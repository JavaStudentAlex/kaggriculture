"""Checkpoint inference / ONNX export. Proposals, NOT a validated game agent.

Only execute proposals after engine-specific legality checks. Replan every turn.
The first future action is returned, alongside the complete categorical chunk.
A conditioned checkpoint (train.py --condition) is asked by default for the play its training
prompt describes (the newest training day, the day's top team, a typical win); guidance > 1
pushes further toward it.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from model import ActionDiffusion, Config, encode_condition, predict


def load(checkpoint):
    """The model, eval mode; `model.conditions` is the checkpoint's condition info (None when
    unconditioned)."""
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = ActionDiffusion(Config(**saved['config']))
    model.load_state_dict(saved['model'])
    model.conditions = saved.get('conditions') if model.config.condition_dim else None
    return model.eval()


def prompt_condition(model, prompt='default', batch=1):
    """The condition tensor asking `model` for one kind of play, or None. prompt: 'default' (the
    checkpoint's training prompt), None (unconditioned) or a dict of model.CONDITIONS values
    (day, rating_gap, win, excess_margin; a missing or None value is unknown)."""
    if prompt is None or not model.config.condition_dim:
        return None
    if prompt == 'default':
        prompt = (model.conditions or {}).get('prompt') or {}
    row = encode_condition(prompt.get('day'), prompt.get('rating_gap'), prompt.get('win'), prompt.get('excess_margin'))
    return torch.tensor([row] * batch, dtype=torch.float32)


def pack_history(observations, seat, context=64):
    from data import encode_observation, FEATURE_DIM
    observations = list(observations)[-context:]
    if not observations:
        raise ValueError('Need at least one observation')
    x = np.zeros((1, context, FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((1, context), dtype=bool)
    x[0, -len(observations):] = np.stack([encode_observation(o, seat) for o in observations])
    mask[0, -len(observations):] = True
    return x, mask


class Forecaster:
    def __init__(self, checkpoint):
        self.model = load(checkpoint)

    def propose(self, observations, seat, steps=1, prompt='default', guidance=1.0):
        """prompt, guidance: see prompt_condition and model.predict (unconditioned models ignore them)."""
        from data import decode_action
        observations = list(observations)
        x, mask = pack_history(observations, seat, self.model.config.context)
        actions, probabilities = predict(self.model, torch.from_numpy(x), torch.from_numpy(mask), steps,
                                         prompt_condition(self.model, prompt), guidance)
        n = len(observations[-1]['farms'][seat].get('hands') or [])
        return dict(action=decode_action(actions[0, 0].numpy(), n),
                    chunk=actions[0].numpy(), probabilities=probabilities[0].numpy(),
                    warning='Uncalibrated behavior forecast; needs engine legality filtering')


def export(checkpoint, out, data):
    torch.set_num_threads(2)
    model = load(checkpoint)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    from data import WindowDataset
    sample = WindowDataset(Path(data), 'val')[0]
    x = torch.as_tensor(sample['context']).float()[None]
    mask = torch.as_tensor(sample['context_mask']).bool()[None]
    actions = model.sizes[None,None,:].expand(1, model.config.horizon, -1)
    t = torch.ones(1)
    model.encoder.enable_nested_tensor = False
    torch.backends.mha.set_fastpath_enabled(False)
    inputs = (x,mask,actions,t)
    names = ['context','context_mask','noisy_actions','noise_fraction']
    if model.config.condition_dim:   # the condition is an input; exported with the default prompt
        inputs += (prompt_condition(model),)
        names.append('condition')
    with torch.no_grad():
        expected = model(*inputs).numpy()
        torch.onnx.export(model, inputs, str(out/'model.onnx'),
                          input_names=names,
                          output_names=['logits'], opset_version=17, dynamo=False)
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(out/'model.onnx'), str(out/'model.int8.onnx'),
                     weight_type=QuantType.QInt8, op_types_to_quantize=['MatMul','Gemm'])
    feed = dict(zip(names,[v.numpy() for v in inputs]))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    results = {}
    for name in ['model.onnx','model.int8.onnx']:
        session = ort.InferenceSession(str(out/name), options, providers=['CPUExecutionProvider'])
        actual = session.run(None,feed)[0]
        valid = np.broadcast_to(model.vocab_mask.numpy(), actual.shape)
        if name == 'model.onnx':
            np.testing.assert_allclose(actual[valid], expected[valid], rtol=1e-3, atol=1e-3)
        for _ in range(3):
            session.run(None,feed)
        times=[]
        for _ in range(20):
            st=time.perf_counter(); session.run(None,feed); times.append((time.perf_counter()-st)*1000)
        results[name] = dict(bytes=(out/name).stat().st_size,
                            median_ms=float(np.median(times)), p95_ms=float(np.percentile(times,95)),
                            max_abs_error=float(np.abs(actual[valid]-expected[valid]).max()),
                            argmax_agreement=float((actual.argmax(-1)==expected.argmax(-1)).mean()))
    results['scope']='CPU of export machine; one validation sample; excludes game parser and legality; not sandbox-certified'
    (out/'export_report.json').write_text(json.dumps(results,indent=2))
    (out/'model_config.json').write_text(json.dumps(model.config_dict(),indent=2))
    print(json.dumps(dict(event='EXPORT_COMPLETE', **results)),flush=True)
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True); p.add_argument('--out',required=True); p.add_argument('--data',required=True)
    a=p.parse_args(); export(a.checkpoint,a.out,a.data)
