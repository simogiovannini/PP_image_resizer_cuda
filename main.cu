#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <cmath>

using namespace cv;
using namespace std;

void sequential_downscale(const Mat& image, int kernel_size);
void cuda_downscale(const Mat& image, int kernel_size);
void save_image(const Mat& image, const string& path);
Vec3b compute_avg_pixel(const Mat& image, int kernel_size, int row_index, int col_index);
__global__ void compute_avg_pixel_cuda(const int* img, int* res, int kernel_size, int img_dim);

int main()
{
    Mat image;
    image = imread("../Lenna.png", IMREAD_COLOR);
    if(image.empty()) { // Check for invalid input
        cout << "Could not open or find the image" << std::endl ;
        return -1;
    }

    cout << "Pixels: " << image.total() << endl;
    cout << image.rows << "x" << image.cols << endl;

    int kernel_size = 8;

    sequential_downscale(image, kernel_size);
    cuda_downscale(image, kernel_size);

    return 0;
}

void sequential_downscale(const Mat& image, int kernel_size) {
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

    save_image(img, "../seq_result.jpg");
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

void cuda_downscale(const Mat& image, int kernel_size) {
    std::vector<int> tmp_array;
    tmp_array.assign(image.data, image.data + image.total()*image.channels());

    int* gpu_img = nullptr;

    cudaMalloc((void**)&gpu_img, image.total() * image.channels() * sizeof(int));
    cudaMemcpy(gpu_img, tmp_array.data(), image.total() * image.channels() * sizeof(int), cudaMemcpyHostToDevice);

    int* new_img = nullptr;
    cudaMalloc((void**)&new_img, image.total() * image.channels() / (kernel_size * kernel_size) * sizeof(int));
    cudaMemset(new_img, 0, image.total() * image.channels() / (kernel_size * kernel_size) * sizeof(int));

    int grid_dim = ceil((float) image.rows / 16);

    dim3 block(16, 16, 3);
    dim3 grid(grid_dim, grid_dim);

    compute_avg_pixel_cuda<<<grid, block>>>(gpu_img, new_img, kernel_size, image.rows);
    cudaDeviceSynchronize();

    int* cuda_img = (int*)malloc(image.total() * image.channels() / (kernel_size * kernel_size) * sizeof(int));
    cudaMemcpy(cuda_img, new_img, image.total() * image.channels() / (kernel_size * kernel_size) * sizeof(int), cudaMemcpyDeviceToHost);

    uchar res[image.total() * image.channels() / (kernel_size * kernel_size)];

    for(int i = 0; i < image.total() * image.channels() / (kernel_size * kernel_size); i += 1) {
        res[i] = cuda_img[i];
    }

    cv::Mat downscaled_img = Mat(image.rows / kernel_size, image.cols / kernel_size, CV_8UC3, &res);
    save_image(downscaled_img, "../cuda_result.jpg");


}

__global__ void compute_avg_pixel_cuda(const int* img, int* res, int kernel_size, int img_dim) {
    int res_dim = img_dim / kernel_size;
    int i =  blockIdx.x * 16 + threadIdx.x;
    int j =  blockIdx.y * 16 + threadIdx.y;
    if(i < img_dim && j < img_dim) {
        int new_i = i / kernel_size;
        int new_j = j / kernel_size;
        int add_val = (img[j * img_dim * 3 + i * 3 + threadIdx.z] / (kernel_size * kernel_size));
        atomicAdd(&res[new_j * res_dim * 3 + new_i * 3 + threadIdx.z], add_val);
    }
}