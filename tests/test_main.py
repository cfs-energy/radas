import pytest
import sys
import importlib.util
from radas.cli import download_data_from_adas, build_raw_datasets

@pytest.fixture()
def datasets(
    monkeypatch,
    temp_module_directory,
    temp_data_file_directory,
    selected_species,
    configuration
): 
    monkeypatch.setattr("radas.adas_interface.download_adas_datasets.data_file_directory", temp_data_file_directory)
    monkeypatch.setattr("radas.adas_interface.prepare_adas_readers.module_directory", temp_module_directory)
    
    download_data_from_adas(configuration)

    assert (temp_data_file_directory / f"{selected_species}_effective_ionisation_coeff.dat").exists()
    assert (temp_module_directory / "readers" / "adf11" / "xxdata_11").exists()

    def mock_read_adf11_file(*args, **kwargs):
        spec = importlib.util.spec_from_file_location("read_adf11_file", temp_module_directory / "readers" / "read_adf11_file.py")
        adf11_module = importlib.util.module_from_spec(spec)
        sys.modules["read_adf11_file"] = adf11_module
        spec.loader.exec_module(adf11_module)

        return adf11_module.read_adf11_file(*args, **kwargs)

    monkeypatch.setattr("radas.build_raw_dataset.read_data_from_adf11_file", mock_read_adf11_file)

    datasets = build_raw_datasets(configuration)

    return datasets

def test_download_data_from_adas_and_build_raw_datasets(datasets):
    pass



# spec = importlib.util.spec_from_file_location("adf11.adf11_reader", temp_module_directory / "readers" / "adf11")


# assert False

# # spec = importlib.util.spec_from_file_location("module.name", "/path/to/file.py")
# # foo = importlib.util.module_from_spec(spec)
# # sys.modules["module.name"] = foo
# # spec.loader.exec_module(foo)
# # foo.MyClass()


# # sys.path.append(temp_module_directory / "readers")
# # sys.path.append(temp_module_directory / "readers" / "adf11")
# # from adf11 import adf11_reader
# # from readers import fortran_file_handling