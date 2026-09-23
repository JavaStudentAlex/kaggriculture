"""Reproduce the small audited engine contract; no runtime Kaggle dependency."""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = Path('/home/alex/kaggriculture-trace-learning/exceptional_audit/engine_1_32_7/kaggriculture.py')
NAMES = {'CROPS', 'PRODUCTS', 'MARKET_I0', 'PRICE_FLOOR', 'MARKET_PARAMS',
         'HINGE_GAIN', 'SHOPS', '_shape', 'market_price'}


def render():
    data = ENGINE.read_bytes()
    source = data.decode()
    selected = []
    for node in ast.parse(source).body:
        names = {t.id for t in node.targets if isinstance(t, ast.Name)} if isinstance(node, ast.Assign) else {getattr(node, 'name', '')}
        if names & NAMES:
            selected.append(ast.get_source_segment(source, node))
    return ('"""Exact selected definitions from pinned Kaggriculture 1.32.7.\n'
            'Regenerate: python docs/build_experimental_contract.py\n'
            f'Engine SHA256: {hashlib.sha256(data).hexdigest()}\n'
            '"""\nimport math\n\n' + '\n\n'.join(selected) + '\n')


if __name__ == '__main__':
    path = ROOT / 'hazel_runtime/engine_contract.py'
    path.write_text(render())
    print(path)
    print(hashlib.sha256(path.read_bytes()).hexdigest())
