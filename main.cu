#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>
#include <cmath>
#include <chrono>
#include <map>

using namespace cv;
using namespace std;

void sequential_downscale(const Mat& image, const string& image_name, int kernel_size);
void cuda_downscale(const Mat& image, const string& image_name, int kernel_size);
void save_image(const Mat& image, const string& path);
Vec3b compute_avg_pixel(const Mat& image, int kernel_size, int row_index, int col_index);
__global__ void compute_avg_pixel_cuda(const int *img, float *res, int kernel_size, int block_dim, int img_dim);

int main() {
    std::chrono::high_resolution_clock::time_point beg, end;
    long long int duration;

    String file_names[] = {"owl.jpeg", "lamborghini.jpg", "mosaic.jpg", "the_last_of_us.jpg", "mushroom.jpg"};
    //String file_names[] = {"owl.jpeg", "lamborghini.jpg", "mosaic.jpg", "the_last_of_us.jpg", "mushroom.jpg"};

    int kernel_sizes[] = {4, 8, 16, 32, 64};

    long long int seq_times[sizeof(kernel_sizes)], par_times[sizeof(kernel_sizes)];

    int n_kernel_sizes = sizeof(kernel_sizes) / sizeof(kernel_sizes)[0];

    for(int i = 0; i < n_kernel_sizes; i++) {
        seq_times[i] = 0;
        par_times[i] = 0;
    }

    for(const auto& file_name : file_names) {
        Mat image;
        String image_path = "../dataset/" + file_name;
        image = imread(image_path, IMREAD_COLOR);
        if(image.empty()) {
            cout << "Could not open or find the image" << std::endl ;
            return -1;
        }

        cout << file_name << ": " << image.rows << "x" << image.cols << endl;

        for(int i = 0; i < n_kernel_sizes; i++) {
            int kernel_size = kernel_sizes[i];
            beg = chrono::high_resolution_clock::now();
            sequential_downscale(image, file_name, kernel_size);
            end = chrono::high_resolution_clock::now();
            duration = chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
            seq_times[i] += duration;

            beg = chrono::high_resolution_clock::now();
            cuda_downscale(image, file_name, kernel_size);
            end = chrono::high_resolution_clock::now();
            duration = chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
            par_times[i] += duration;
        }
    }

    for(int i = 0; i < n_kernel_sizes; i++){
        cout << endl << "Kernel size: " << kernel_sizes[i] << endl;
        cout << "Average elapsed time for sequential downscale: " << (float) seq_times[i] / n_kernel_sizes << endl;
        cout << "Average elapsed time for parallelized downscale: " << (float) par_times[i] / n_kernel_sizes  << endl;
    }

    return 0;
}

void sequential_downscale(const Mat& image, const string& image_name, int kernel_size) {
    int rows = image.rows;
    int cols = image.cols;

    Mat img(rows / kernel_size, cols / kernel_size, CV_8UC3, Scalar(0,0, 0));
    int newImgRow = 0;
    int newImgCol;

    for(int i = 0; i < rows; i += kernel_size) {
        newImgRow++;
        newImgCol = 0;
        for(int j = 0; j < cols; j += kernel_size) {
            Vec3b avgPixel = compute_avg_pixel(image, kernel_size, i, j);
            img.at<Vec3b>(Point(newImgRow, newImgCol)) = avgPixel;
            newImgCol++;
        }
    }

    String save_path = "../seq_result/" + to_string(kernel_size) + "_" + image_name;
    save_image(img, save_path);
}

void save_image(const Mat& image, const string& path) {
    imwrite(path, image);
}

Vec3b compute_avg_pixel(const Mat& image, int kernel_size, int row_index, int col_index) {
    Vec3b avg;
    int sumB = 0;
    int sumG = 0;
    int sumR = 0;
    for(int i = row_index; i < row_index + kernel_size; i++) {
        for(int j = col_index; j < col_index + kernel_size; j++) {
            Vec3b tmp = image.at<Vec3b>(Point(i, j));
            sumB += tmp[0];
            sumG += tmp[1];
            sumR += tmp[2];
        }
    }
    int n_pixel_kernel = kernel_size * kernel_size;
    avg[0] = (int) sumB / n_pixel_kernel;
    avg[1] = (int) sumG / n_pixel_kernel;
    avg[2] = (int) sumR / n_pixel_kernel;
    return avg;
}

void cuda_downscale(const Mat& image, const string& image_name, int kernel_size) {
    std::vector<int> tmp_array;
    tmp_array.assign(image.data, image.data + image.total()*image.channels());

    int* gpu_img = nullptr;

    cudaMalloc((void**)&gpu_img, image.total() * image.channels() * sizeof(int));
    cudaMemcpy(gpu_img, tmp_array.data(), image.total() * image.channels() * sizeof(int), cudaMemcpyHostToDevice);

    float* new_img = nullptr;
    cudaMalloc((void**)&new_img, image.total() * image.channels() / (kernel_size * kernel_size) * sizeof(float));
    cudaMemset(new_img, 0, image.total() * image.channels() / (kernel_size * kernel_size) * sizeof(float));

    int block_dim = 16;
    int grid_dim = ceil((float) image.rows / block_dim);
    dim3 block(block_dim, block_dim, 3);
    dim3 grid(grid_dim, grid_dim);
    compute_avg_pixel_cuda<<<grid, block>>>(gpu_img, new_img, kernel_size, block_dim, image.rows);
    cudaDeviceSynchronize();

    float* cuda_img = (float*)malloc(image.total() * image.channels() / (kernel_size * kernel_size) * sizeof(float));
    cudaMemcpy(cuda_img, new_img, image.total() * image.channels() / (kernel_size * kernel_size) * sizeof(float), cudaMemcpyDeviceToHost);

    uchar res[image.total() * image.channels() / (kernel_size * kernel_size)];

    for(int i = 0; i < image.total() * image.channels() / (kernel_size * kernel_size); i += 1) {
        res[i] = cuda_img[i];
    }

    cv::Mat downscaled_img = Mat(image.rows / kernel_size, image.cols / kernel_size, CV_8UC3, &res);

    String save_path = "../cuda_result/" + to_string(kernel_size) + "_" + image_name;
    save_image(downscaled_img, save_path);
}

__global__ void compute_avg_pixel_cuda(const int *img, float *res, int kernel_size, int block_dim, int img_dim) {
    int res_dim = img_dim / kernel_size;
    int i =  blockIdx.x * block_dim + threadIdx.x;
    int j =  blockIdx.y * block_dim + threadIdx.y;
    if(i < img_dim && j < img_dim) {
        int new_i = i / kernel_size;
        int new_j = j / kernel_size;
        float pixel_val = img[j * img_dim * 3 + i * 3 + threadIdx.z];
        float add_val = pixel_val / (float) (kernel_size * kernel_size);
        atomicAdd(&res[new_j * res_dim * 3 + new_i * 3 + threadIdx.z], add_val);
    }
}