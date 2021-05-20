#include <fstream>
#include <iostream>
#include <string>
#include "mpi.h"

int main(int argc, char* argv[]) {
    MPI_Status status;
    MPI_Init(&argc, &argv);

    int rank, size, n_rows, n_rows_local, n_cols, n_iter;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int root_rank = 0;

    // один процесс считывает файл с исходным состоянием и обрабатывает его
    if (rank == root_rank) {
        if (argc != 2) {
            std::cerr << "Input file is not specified!";
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }

        std::ifstream input_file(argv[1]);
        if (!input_file) {
            std::cerr << "Unable to open file: " << argv[1];
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }

        // считываение параметров из первой строки
        input_file >> n_rows >> n_cols >> n_iter;

        // считывание исходного состояния игры
        int grid[n_rows][n_cols];
        for (int i = 0; i < n_rows; ++i){
            std::string row;
            input_file >> row;

            for (int j = 0; j < n_cols; ++j) {
                grid[i][j] = row[j] - '0';
            }
        }
        input_file.close();

        // каждому процессу отправляем информацию о части поля, которую он будет обрабатывать
        int grid_info[3];
        // и исходные значения клеток в этой части поля
        int grid_row[n_cols];

        for (int dest_rank = 0; dest_rank < size; ++dest_rank) {
            // число строк поля, которое будет обрабатывать процесс
            n_rows_local = n_rows / size;
            // если нельзя разделить строки поровну, то остаток будет обрабатывать последний процесс
            if (dest_rank == size - 1){
                n_rows_local += n_rows % size;
            }

            grid_info[0] = n_rows_local;
            grid_info[1] = n_cols;
            grid_info[2] = n_iter;

            MPI_Send(&grid_info, 3, MPI_INT, dest_rank, 1, MPI_COMM_WORLD);

            int skip_rows = (n_rows / size) * dest_rank;
            for (int i = 0; i < n_rows_local; ++i) {
                for (int j = 0; j < n_cols; ++j) {
                    grid_row[j] = grid[skip_rows + i][j];
                }

                MPI_Send(&grid_row, n_cols, MPI_INT, dest_rank, 1, MPI_COMM_WORLD);
            }
        }
    }

    const int ALIVE = 1;
    const int DEAD = 0;

    // принимаем данные о части поля, которую надо будет обрабатывать
    int local_grid_info[3];
    MPI_Recv(&local_grid_info, 3, MPI_INT, root_rank, 1, MPI_COMM_WORLD, &status);
    n_rows_local = local_grid_info[0];
    n_cols = local_grid_info[1];
    n_iter = local_grid_info[2];

    // в поле определим фиктивные строки сверху и снизу и столбцы слева и справа,
    // чтобы удобнее было обрабатывать клетки по краю поля
    int n_rows_local_with_pad = n_rows_local + 2;
    int n_cols_with_pad = n_cols + 2;

    // массив с текущим состоянием поля
    int local_grid_slice[n_rows_local_with_pad][n_cols_with_pad];
    // массив для промежуточного вычисления следующего состояния поля
    int next_local_grid_slice[n_rows_local_with_pad][n_cols_with_pad];

    // получаем по очереди значения клеток в строках
    for (int i = 1; i < n_rows_local + 1; ++i) {
        MPI_Recv(&local_grid_slice[i][1], n_cols, MPI_INT, root_rank, 1, MPI_COMM_WORLD, &status);
    }

    // определяем, с какими процесами надо будет обмениваться граничными значениями клеток
    int up_rank = rank == 0 ? size - 1 : rank - 1;
    int down_rank = rank == size - 1 ? 0 : rank + 1;

    // выполняем указанное число итераций игры
    for (int iter = 0; iter < n_iter; ++iter) {
        // отправляем первую строку своего поля вверх
        MPI_Send(&local_grid_slice[1][0], n_cols_with_pad, MPI_INT, up_rank, 1, MPI_COMM_WORLD);
        // отправляем последнюю строку своего поля вниз
        MPI_Send(&local_grid_slice[n_rows_local][0], n_cols_with_pad, MPI_INT, down_rank, 1, MPI_COMM_WORLD);

        // принимаем соответствующие строки от своих соседей
		// нижнего
        MPI_Recv(&local_grid_slice[n_rows_local + 1][0], n_cols_with_pad, MPI_INT, down_rank, 1, MPI_COMM_WORLD, &status);
		// и верхнего
        MPI_Recv(&local_grid_slice[0][0], n_cols_with_pad, MPI_INT, up_rank, 1, MPI_COMM_WORLD, &status);

        // инициализируем фиктивные колонки, чтобы зациклить поле
        for (int i = 0; i < n_rows_local_with_pad; ++i) {
            local_grid_slice[i][0] = local_grid_slice[i][n_cols];
            local_grid_slice[i][n_cols + 1] = local_grid_slice[i][1];
        }

        // обрабатываем каждую клетку по очереди
        for (int i = 1; i < n_rows_local + 1; ++i) {
            for (int j = 1; j < n_cols + 1; ++j) {
                // подсчитываем число живых соседей
                int n_alive_neighbours = 0;

                for (int di = -1; di < 2; ++di) {
                    for (int dj = -1; dj < 2; ++dj) {
                        int cell = local_grid_slice[i + di][j + dj];

                        if ((di != 0 || dj != 0) && cell == ALIVE) {
                            ++n_alive_neighbours;
                        }
                    }
                }

                // определяем новое значение клетки в соответствии с правилами
                if (local_grid_slice[i][j] == ALIVE) {
                    if (n_alive_neighbours == 2 || n_alive_neighbours == 3) {
                        next_local_grid_slice[i][j] = ALIVE;
                    } else {
                        next_local_grid_slice[i][j] = DEAD;
                    }
                }

                if (local_grid_slice[i][j] == DEAD) {
                    if (n_alive_neighbours == 3) {
                        next_local_grid_slice[i][j] = ALIVE;
                    } else {
                        next_local_grid_slice[i][j] = DEAD;
                    }
                }
            }
        }

        // обновляем текущее состояние доски
        for (int i = 1; i < n_rows_local + 1; ++i) {
            for (int j = 1; j < n_cols + 1; ++j) {
                local_grid_slice[i][j] = next_local_grid_slice[i][j];
            }
        }
    }

    // отсылаем итоговые значения клеток одному процессу
    if (rank != root_rank) {
        for (int i = 1; i < n_rows_local + 1; ++ i) {
            MPI_Send(&local_grid_slice[i][1], n_cols, MPI_INT, root_rank, 1, MPI_COMM_WORLD);

        }
    }

    // принимаем значения всех клеток от процессов и записываем в выходной файл
    if (rank == root_rank) {
        std::ofstream output_file("output.txt");

        for (int i = 1; i < n_rows_local + 1; ++i) {
            for (int j = 1; j < n_cols + 1; ++j) {
                output_file << local_grid_slice[i][j];
            }
            output_file << std::endl;
        }

        int grid_row[n_cols];
        for (int source_rank = 1; source_rank < size; ++source_rank) {
            int source_rows = n_rows / size;
            if (source_rank == size - 1) {
                source_rows += n_rows % size;
            }

            for (int i = 0; i < source_rows; ++i) {
                MPI_Recv(&grid_row, n_cols, MPI_INT, source_rank, 1, MPI_COMM_WORLD, &status);
                for (int j = 0; j < n_cols; ++j) {
                    output_file << grid_row[j];
                }
                output_file << std::endl;
            }
        }
        output_file.close();
    }

    MPI_Finalize();
    return 0;
}
