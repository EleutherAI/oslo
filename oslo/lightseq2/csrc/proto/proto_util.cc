#include "proto_util.h"

namespace lightseq {

bool endswith(std::string const &full, std::string const &end) {
  if (full.length() >= end.length()) {
    return (0 == full.compare(full.length() - end.length(), end.length(), end));
  }
  return false;
}

int get_hdf5_dataset_size(hid_t dataset) {
  hid_t dataspace = H5Dget_space(dataset); /* dataspace handle */
  int n_dims = H5Sget_simple_extent_ndims(dataspace);
  // return 1 for scalar
  if (n_dims < 1) {
    return 1;
  }
  // get dimensions for N-Dimension vector
  hsize_t dims[n_dims];
  int status = H5Sget_simple_extent_dims(dataspace, dims, NULL);
  if (status != n_dims || status < 0) {
    // return negative number on error
    return -1;
  }
  // accumulate size from every dimension
  int vec_size = 1;
  for (int i = 0; i < n_dims; ++i) {
    vec_size *= dims[i];
  }
  return vec_size;
}

int get_hdf5_dataset_size(hid_t hdf5_file, std::string dataset_name) {
  // check if dataset exists or not
  if (!H5Lexists(hdf5_file, dataset_name.c_str(), H5P_DEFAULT)) {
    throw HDF5DatasetNotFoundError(
        (dataset_name + " Not Found in HDF5 File").c_str());
  }

  // parse dataset size
  hid_t ds = H5Dopen2(hdf5_file, dataset_name.c_str(), H5P_DEFAULT);
  if (ds < 0) {
    throw std::runtime_error("Failed to open HDF5 dataset: " + dataset_name);
  }
  int ds_size = get_hdf5_dataset_size(ds);
  if (ds_size < 0) {
    throw std::runtime_error("HDF5 parsing error: " + dataset_name);
  }
  H5Dclose(ds);
  return ds_size;
}

int read_hdf5_dataset_data(hid_t hdf5_file, std::string dataset_name,
                           hid_t output_type, void *output_buf,
                           std::function<bool(int)> size_predicate,
                           std::string extra_msg) {
  // check if dataset exists or not
  if (!H5Lexists(hdf5_file, dataset_name.c_str(), H5P_DEFAULT)) {
    throw HDF5DatasetNotFoundError(
        (dataset_name + " Not Found in HDF5 File").c_str());
  }

  hid_t ds = H5Dopen2(hdf5_file, dataset_name.c_str(), H5P_DEFAULT);
  if (ds < 0) {
    throw std::runtime_error("Failed to open HDF5 dataset: " + dataset_name);
  }
  int ds_size = get_hdf5_dataset_size(ds);

  // sanity (custom) check for size with extra message.
  if (size_predicate(ds_size)) {
    throw std::runtime_error("Invalid shape " + std::to_string(ds_size) + ". " +
                             extra_msg);
  }

  herr_t status =
      H5Dread(ds, output_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, output_buf);

  if (status < 0) {
    throw std::runtime_error("Failed to read HDF5 dataset: " + dataset_name);
  }
  H5Dclose(ds);
  return ds_size;
}

std::vector<float> read_hdf5_dataset_data_float(
    hid_t hdf5_file, std::string dataset_name, hid_t output_type,
    std::function<bool(int)> size_predicate, std::string extra_msg) {
  // check if dataset exists or not
  if (!H5Lexists(hdf5_file, dataset_name.c_str(), H5P_DEFAULT)) {
    throw HDF5DatasetNotFoundError(
        (dataset_name + " Not Found in HDF5 File").c_str());
  }

  hid_t ds = H5Dopen2(hdf5_file, dataset_name.c_str(), H5P_DEFAULT);
  if (ds < 0) {
    throw std::runtime_error("Failed to open HDF5 dataset: " + dataset_name);
  }
  int ds_size = get_hdf5_dataset_size(ds);

  // sanity (custom) check for size with extra message.
  if (size_predicate(ds_size)) {
    throw std::runtime_error("Invalid shape " + std::to_string(ds_size) + ". " +
                             extra_msg);
  }

  std::vector<float> output_vec(ds_size);
  herr_t status = H5Dread(ds, output_type, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                          output_vec.data());

  if (status < 0) {
    throw std::runtime_error("Failed to read HDF5 dataset: " + dataset_name);
  }
  H5Dclose(ds);
  return output_vec; // return with copy elision
}

std::vector<int> read_hdf5_dataset_data_int(
    hid_t hdf5_file, std::string dataset_name, hid_t output_type,
    std::function<bool(int)> size_predicate, std::string extra_msg) {
  // check if dataset exists or not
  if (!H5Lexists(hdf5_file, dataset_name.c_str(), H5P_DEFAULT)) {
    throw HDF5DatasetNotFoundError(
        (dataset_name + " Not Found in HDF5 File").c_str());
  }

  hid_t ds = H5Dopen2(hdf5_file, dataset_name.c_str(), H5P_DEFAULT);
  if (ds < 0) {
    throw std::runtime_error("Failed to open HDF5 dataset: " + dataset_name);
  }
  int ds_size = get_hdf5_dataset_size(ds);

  // sanity (custom) check for size with extra message.
  if (size_predicate(ds_size)) {
    throw std::runtime_error("Invalid shape " + std::to_string(ds_size) + ". " +
                             extra_msg);
  }

  std::vector<int> output_vec(ds_size);
  herr_t status = H5Dread(ds, output_type, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                          output_vec.data());

  if (status < 0) {
    throw std::runtime_error("Failed to read HDF5 dataset: " + dataset_name);
  }
  H5Dclose(ds);
  return output_vec; // return with copy elision
}

int read_hdf5_dataset_scalar(hid_t hdf5_file, std::string dataset_name,
                             hid_t output_type, void *output_buf) {
  return read_hdf5_dataset_data(
      hdf5_file, dataset_name, output_type, output_buf,
      [](int size) { return size != 1; }, "Expect scalar with shape of 1.");
}

void transform_param_shape(float *origin, float *buffer, int row_size,
                           int col_size) {
  int idx = 0;
  for (int i = 0; i < row_size; i++) {
    for (int j = 0; j < col_size; j++) {
      *(buffer + j * row_size + i) = *(origin + idx);
      idx++;
    }
  }
  for (int i = 0; i < row_size * col_size; i++) {
    *(origin + i) = *(buffer + i);
  }
}
} // namespace lightseq
