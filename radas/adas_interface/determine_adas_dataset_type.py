def determine_reader_class_and_config(data_file_config, dataset_type):
    """Examines the data_file_config to determine which reader class to use to read a specific dataset_type."""
    for reader_key, reader_config in data_file_config.items():
        for dataset_key, dataset_config in reader_config.items():
            if dataset_key == dataset_type:
                return reader_key, dataset_config

    raise NotImplementedError(f"Cannot identify reader for {dataset_type}.")
