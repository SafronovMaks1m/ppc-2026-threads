#include "safronov_m_multiplication_matrix_blockscheme_cannon/all/include/ops_all.hpp"

#include <mpi.h>

#include <cmath>
#include <utility>

#include "oneapi/tbb/blocked_range2d.h"
#include "oneapi/tbb/parallel_for.h"

namespace safronov_m_multiplication_matrix_blocksscheme_cannon {

SafronovMMultiplicationMatrixBlockSchemeCannonALL::SafronovMMultiplicationMatrixBlockSchemeCannonALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SafronovMMultiplicationMatrixBlockSchemeCannonALL::ValidationImpl() {
  const auto &in = GetInput();
  int size_block = std::get<0>(in);
  const auto &matrix_a = std::get<1>(in);
  const auto &matrix_b = std::get<2>(in);

  return (size_block > 0) && (!matrix_a.empty() && !matrix_b.empty()) && (matrix_a.size() == matrix_a[0].size()) &&
         (matrix_b.size() == matrix_b[0].size()) && (matrix_a.size() == matrix_b.size());
}

bool SafronovMMultiplicationMatrixBlockSchemeCannonALL::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

int SafronovMMultiplicationMatrixBlockSchemeCannonALL::CalcPaddedSize(int n, int q) {
  return ((n + q - 1) / q) * q;
}

void SafronovMMultiplicationMatrixBlockSchemeCannonALL::PadMatrix(const std::vector<std::vector<double>> &src,
                                                                  std::vector<std::vector<double>> &dst, int padded_n) {
  dst.assign(static_cast<size_t>(padded_n), std::vector<double>(static_cast<size_t>(padded_n), 0.0));

  int n = static_cast<int>(src.size());
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      dst[i][j] = src[i][j];
    }
  }
}

void SafronovMMultiplicationMatrixBlockSchemeCannonALL::ParallelMultiplyBlocks(const std::vector<double> &A,
                                                                               const std::vector<double> &B,
                                                                               std::vector<double> &C, int block_size) {
  tbb::parallel_for(tbb::blocked_range2d<int>(0, block_size, 0, block_size), [&](const tbb::blocked_range2d<int> &r) {
    for (int i = r.rows().begin(); i < r.rows().end(); ++i) {
      for (int k = 0; k < block_size; ++k) {
        double temp = A[i * block_size + k];
        for (int j = r.cols().begin(); j < r.cols().end(); ++j) {
          C[i * block_size + j] += temp * B[k * block_size + j];
        }
      }
    }
  });
}

void SafronovMMultiplicationMatrixBlockSchemeCannonALL::DistributeData(
    MPI_Comm comm, int worker_rank, int worker_size, int q, int block_size,
    const std::vector<std::vector<double>> &matrix_a_full, const std::vector<std::vector<double>> &matrix_b_full,
    std::vector<double> &local_A, std::vector<double> &local_B) {
  if (worker_rank == 0) {
    for (int p = 0; p < worker_size; ++p) {
      int p_row = p / q;
      int p_col = p % q;

      std::vector<double> send_A(static_cast<size_t>(block_size) * block_size);
      std::vector<double> send_B(static_cast<size_t>(block_size) * block_size);

      for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < block_size; ++j) {
          int a_row = p_row * block_size + i;
          int a_col = ((p_col + p_row) % q) * block_size + j;
          int b_row = ((p_row + p_col) % q) * block_size + i;
          int b_col = p_col * block_size + j;

          send_A[i * block_size + j] = matrix_a_full[a_row][a_col];
          send_B[i * block_size + j] = matrix_b_full[b_row][b_col];
        }
      }

      if (p == 0) {
        local_A = std::move(send_A);
        local_B = std::move(send_B);
      } else {
        MPI_Send(send_A.data(), block_size * block_size, MPI_DOUBLE, p, 0, comm);
        MPI_Send(send_B.data(), block_size * block_size, MPI_DOUBLE, p, 1, comm);
      }
    }
  } else {
    MPI_Recv(local_A.data(), block_size * block_size, MPI_DOUBLE, 0, 0, comm, MPI_STATUS_IGNORE);
    MPI_Recv(local_B.data(), block_size * block_size, MPI_DOUBLE, 0, 1, comm, MPI_STATUS_IGNORE);
  }
}

void SafronovMMultiplicationMatrixBlockSchemeCannonALL::CannonAlgorithm(MPI_Comm comm, int worker_rank, int q,
                                                                        int block_size, std::vector<double> &local_A,
                                                                        std::vector<double> &local_B,
                                                                        std::vector<double> &local_C) {
  int row = worker_rank / q;
  int col = worker_rank % q;

  int left = row * q + (col - 1 + q) % q;
  int right = row * q + (col + 1) % q;
  int up = ((row - 1 + q) % q) * q + col;
  int down = ((row + 1) % q) * q + col;

  for (int step = 0; step < q; ++step) {
    ParallelMultiplyBlocks(local_A, local_B, local_C, block_size);

    if (step < q - 1) {
      std::vector<double> next_A(static_cast<size_t>(block_size) * block_size);
      std::vector<double> next_B(static_cast<size_t>(block_size) * block_size);

      MPI_Sendrecv(local_A.data(), block_size * block_size, MPI_DOUBLE, left, 10, next_A.data(),
                   block_size * block_size, MPI_DOUBLE, right, 10, comm, MPI_STATUS_IGNORE);

      MPI_Sendrecv(local_B.data(), block_size * block_size, MPI_DOUBLE, up, 11, next_B.data(), block_size * block_size,
                   MPI_DOUBLE, down, 11, comm, MPI_STATUS_IGNORE);

      local_A = std::move(next_A);
      local_B = std::move(next_B);
    }
  }
}

void SafronovMMultiplicationMatrixBlockSchemeCannonALL::CollectResult(MPI_Comm comm, int worker_rank, int worker_size,
                                                                      int q, int block_size,
                                                                      std::vector<double> &flat_result,
                                                                      const std::vector<double> &local_C) {
  const int padded_n = q * block_size;

  if (worker_rank == 0) {
    for (int i = 0; i < block_size; ++i) {
      for (int j = 0; j < block_size; ++j) {
        flat_result[i * padded_n + j] = local_C[i * block_size + j];
      }
    }

    std::vector<double> recv_buf(static_cast<size_t>(block_size) * block_size);
    for (int p = 1; p < worker_size; ++p) {
      MPI_Recv(recv_buf.data(), block_size * block_size, MPI_DOUBLE, p, 20, comm, MPI_STATUS_IGNORE);

      int p_row = p / q;
      int p_col = p % q;

      for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < block_size; ++j) {
          int row = p_row * block_size + i;
          int col = p_col * block_size + j;
          flat_result[row * padded_n + col] = recv_buf[i * block_size + j];
        }
      }
    }
  } else {
    MPI_Send(local_C.data(), block_size * block_size, MPI_DOUBLE, 0, 20, comm);
  }
}

bool SafronovMMultiplicationMatrixBlockSchemeCannonALL::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int q = static_cast<int>(std::floor(std::sqrt(size)));
  int working_proc_count = q * q;

  int original_n = 0;
  if (rank == 0) {
    original_n = static_cast<int>(std::get<1>(GetInput()).size());
  }
  MPI_Bcast(&original_n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int padded_n = CalcPaddedSize(original_n, std::max(1, q));
  int block_size = padded_n / std::max(1, q);

  std::vector<std::vector<double>> padded_a;
  std::vector<std::vector<double>> padded_b;

  if (rank == 0) {
    PadMatrix(std::get<1>(GetInput()), padded_a, padded_n);
    PadMatrix(std::get<2>(GetInput()), padded_b, padded_n);
  }

  std::vector<double> flat_result(static_cast<size_t>(padded_n) * padded_n, 0.0);

  MPI_Comm cannon_comm = MPI_COMM_NULL;
  int color = (rank < working_proc_count) ? 0 : MPI_UNDEFINED;
  MPI_Comm_split(MPI_COMM_WORLD, color, rank, &cannon_comm);

  if (rank < working_proc_count) {
    int worker_rank = 0;
    int worker_size = 0;
    MPI_Comm_rank(cannon_comm, &worker_rank);
    MPI_Comm_size(cannon_comm, &worker_size);

    std::vector<double> local_A(static_cast<size_t>(block_size) * block_size);
    std::vector<double> local_B(static_cast<size_t>(block_size) * block_size);
    std::vector<double> local_C(static_cast<size_t>(block_size) * block_size, 0.0);

    DistributeData(cannon_comm, worker_rank, worker_size, q, block_size, padded_a, padded_b, local_A, local_B);

    CannonAlgorithm(cannon_comm, worker_rank, q, block_size, local_A, local_B, local_C);

    CollectResult(cannon_comm, worker_rank, worker_size, q, block_size, flat_result, local_C);

    MPI_Comm_free(&cannon_comm);
  }

  MPI_Bcast(flat_result.data(), padded_n * padded_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<std::vector<double>> final_matrix(static_cast<size_t>(original_n),
                                                std::vector<double>(static_cast<size_t>(original_n)));

  for (int i = 0; i < original_n; ++i) {
    for (int j = 0; j < original_n; ++j) {
      final_matrix[i][j] = flat_result[i * padded_n + j];
    }
  }

  GetOutput() = std::move(final_matrix);

  return true;
}

bool SafronovMMultiplicationMatrixBlockSchemeCannonALL::PostProcessingImpl() {
  return true;
}

}  // namespace safronov_m_multiplication_matrix_blocksscheme_cannon
