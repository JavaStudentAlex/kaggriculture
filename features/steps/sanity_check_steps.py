from behave import given, then, when

from sanity_check import main


@given("repository environment is initialized")
def step_impl_given(context):
    context.ready = True


@when("sanity check function is called")
def step_impl_when(context):
    context.result = main()


@then("result is zero")
def step_impl_then(context):
    assert context.result == 0
