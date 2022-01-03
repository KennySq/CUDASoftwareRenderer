#include<pch.h>
#include"Renderer.cuh"
#include"DIB.cuh"
#include"DeviceTexture.cuh"
#include"ResourceManager.cuh"
#include"Geometry.cuh"
#include"3DMath.cuh"
#include"Util.h"

__device__ Renderer::Point2D* deviceDrawPoints = nullptr;
__device__ Renderer::Triangle2D* deviceTriangles = nullptr;

__global__ void KernelClearBitmap(void* target, unsigned int width, unsigned int height, ColorRGBA clearColor)
{
	unsigned int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	DWORD* asPixel = reinterpret_cast<DWORD*>(target);

	asPixel[index] = ConvertColorToDWORD(clearColor);
}

Renderer::Renderer(std::shared_ptr<DIB> dib, std::shared_ptr<ResourceManager> rs)
	: mCanvas(dib)
{
	unsigned int width = mCanvas->GetWidth();
	unsigned int height = mCanvas->GetHeight(); 

	mBuffer = rs->CreateTexture2D(width, height);

	mRenderPoints = new Point2D[width * height];
	mPointCount = 0;
	cudaError_t error = cudaMalloc(reinterpret_cast<void**>(&deviceDrawPoints), width * height * sizeof(Point2D));
	CUDAError(error);

	mRenderTriangles.resize(1024);

}

Renderer::~Renderer()
{
	Release();
}

void Renderer::SetPixel(int x, int y, const ColorRGBA& color)
{
	unsigned int width = mCanvas->GetWidth();
	unsigned int height = mCanvas->GetHeight();
	if (mPointCount > width * height)
	{
		return;
	}

	unsigned int _x = (width / 2 - x) - 1;
	unsigned int _y = (height / 2 - y) - 1;
	mRenderPoints[mPointCount] = Point2D(INT2(_x, _y), color);

	mPointCount++;
}

void Renderer::SetPixelNDC(float x, float y, const ColorRGBA& color)
{
	unsigned int width = mCanvas->GetWidth();
	unsigned int height = mCanvas->GetHeight();
	if (mPointCount > width * height)
	{
		return;
	}

	unsigned int _x = (width / 2 + (x * width)) - 1;
	unsigned int _y = (height / 2 + (y * height)) - 1;
	mRenderPoints[mPointCount] = Point2D(INT2(_x, _y), color);

	mPointCount++;
}

void Renderer::SetTriangle(const Point2D& p0, const Point2D& p1, const Point2D& p2)
{


	return;
}

void Renderer::Start()
{

	unsigned int width = mCanvas->GetWidth();
	unsigned int height = mCanvas->GetHeight();

	cudaError_t error = cudaMalloc(reinterpret_cast<void**>(&deviceDrawPoints), width * height * sizeof(Point2D));
	CUDAError(error);

	error = cudaMalloc(reinterpret_cast<void**>(&deviceTriangles), sizeof(Triangle2D) * mTriangleCount);
	CUDAError(error);


	error = cudaMemcpy(deviceTriangles, mRenderTriangles.data(), mRenderTriangles.size() * sizeof(Triangle2D), cudaMemcpyHostToDevice);
	CUDAError(error);
}

void Renderer::Update()
{
}

void Renderer::Render()
{
}

void Renderer::Release()
{
	cudaFree(deviceDrawPoints);
	cudaFree(deviceTriangles);
	delete[] mRenderPoints;
	mRenderPoints = nullptr;
}

void Renderer::ClearCanvas(const ColorRGBA& clearColor)
{
	unsigned int width = mCanvas->GetWidth();
	unsigned int height = mCanvas->GetHeight();

	void* texture = mBuffer->GetVirtual();

	dim3 block = dim3(32, 18, 1);
	dim3 grid = dim3(width / block.x, height / block.y, 1);

	if (ColorRGBA(0, 0, 0, 0) == clearColor)
	{
		return;
	}

	KernelClearBitmap<<<grid, block>>>(texture, width, height, clearColor);

	cudaDeviceSynchronize();
}

void Renderer::Present()
{
	mCanvas->CopyBuffer(mBuffer);
	mCanvas->Present();
}

inline __device__ void DeviceSetPixel(DWORD* buffer, unsigned int pointIndex, const ColorRGBA& color, unsigned int width)
{
	buffer[pointIndex] = ConvertColorToDWORD(color);
}

__global__ void KernelDrawCallSetLine(DWORD* buffer, Renderer::Line2D* drawLines)
{
	unsigned int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	

}

__global__ void KernelTransformVertices(DWORD* buffer, SampleVertex* vertices, unsigned int* indices, unsigned int vertexCount, unsigned int indexCount, const FLOAT4X4& Transform, const FLOAT4X4& View, const FLOAT4X4& Projection)
{
	unsigned int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	unsigned int triIndex = index * 3;

	if (triIndex + 2 >= vertexCount)
	{
		return;
	}

	SampleVertex v0 = vertices[triIndex];
	SampleVertex v1 = vertices[triIndex + 1];
	SampleVertex v2 = vertices[triIndex + 2];
	
	return;
}



__global__ void KernelDrawCallSetPixel(DWORD* buffer, Renderer::Point2D* drawPoints, unsigned int pixelCount, unsigned int width)
{
	unsigned int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	Renderer::Point2D pixel = drawPoints[index];
	unsigned int pointIndex = (pixel.Point.y * width) + pixel.Point.x;

	if (pointIndex >= pixelCount)
	{
		return;
	}

	buffer[pointIndex] = ConvertColorToDWORD(pixel.Color);

}



void Renderer::DrawScreen()
{
	unsigned int width = mCanvas->GetWidth();
	unsigned int height = mCanvas->GetHeight();

	dim3 block = dim3(32, 18, 1);
	dim3 grid = dim3(width / block.x, height / block.y, 1);
	
	void* buffer = mBuffer->GetVirtual();

	size_t copySize = width * height * sizeof(Point2D);
	cudaError_t error = cudaMemcpy(deviceDrawPoints, mRenderPoints, copySize, cudaMemcpyHostToDevice);
	CUDAError(error);

	cudaDeviceSynchronize();

	KernelDrawCallSetPixel << <grid, block >> > (CAST_PIXEL(buffer), deviceDrawPoints, width * height, width);

	cudaDeviceSynchronize();
}

void Renderer::DrawTriangles()
{
	unsigned int width = mCanvas->GetWidth();
	unsigned int height = mCanvas->GetHeight();

	void* buffer = mBuffer->GetVirtual();

	dim3 block = dim3(32, 1, 1);
	int left = mTriangleCount % 3;
	dim3 grid = dim3((mTriangleCount / block.x) + left, 1,1);

	//KernelDrawCallSetTriangle<<<grid, block>>>(CAST_PIXEL(buffer), mTriangleCount, deviceTriangles, width);

	cudaDeviceSynchronize();
}
