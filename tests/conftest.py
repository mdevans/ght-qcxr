import pytest
from segment import load_cxas_model, load_xrv_model, DEVICE

@pytest.fixture(scope="session")
def loaded_models():
    """
    Loads models ONCE per test session. 
    Shared automatically across test_segment, test_ensemble, and test_exposure.
    """
    print(f"\n🚀 [conftest.py] Loading Real Models into {DEVICE} memory for entire test suite...")
    return load_cxas_model(DEVICE), load_xrv_model(DEVICE)