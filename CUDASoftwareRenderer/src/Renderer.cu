#include<pch.h>
#include"Renderer.cuh"
#include"DIB.cuh"
#include"DeviceTexture.cuh"
#include"DeviceBuffer.cuh"
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

__global__ void KernelTransformVertices(SampleVertex* vertices, VertexOutput* output, unsigned int* indices, unsigned int vertexCount, unsigned int indexCount, FLOAT4X4 Transform, FLOAT4X4 View, FLOAT4X4 Projection)
{
	unsigned int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	unsigned int triThread = index * 3;

	if (triThread + 2 >= vertexCount)
	{
		return;
	}

	unsigned int triIndex0 = indices[triThread];
	unsigned int triIndex1 = indices[triThread + 1];
	unsigned int triIndex2 = indices[triThread + 2];

	SampleVertex v0 = vertices[triIndex0];
	SampleVertex v1 = vertices[triIndex1];
	SampleVertex v2 = vertices[triIndex2];
	
	FLOAT4 position0 = FLOAT4(v0.Position.x, v0.Position.y, v0.Position.z, 1.0f);
	FLOAT4 position1 = FLOAT4(v1.Position.x, v1.Position.y, v1.Position.z, 1.0f);
	FLOAT4 position2 = FLOAT4(v2.Position.x, v2.Position.y, v2.Position.z, 1.0f);



	position0 = Float4Multiply(position0, Transform);
	position0 = Float4Multiply(position0, View);
	position0 = Float4Multiply(position0, Projection);
	
	position1 = Float4Multiply(position1, Transform);
	position1 = Float4Multiply(position1, View);
	position1 = Float4Multiply(position1, Projection);
	
	position2 = Float4Multiply(position2, Transform);
	position2 = Float4Multiply(position2, View);
	position2 = Float4Multiply(position2, Projection);

	FLOAT4 normal0 = FLOAT4(v0.Normal.x, v0.Normal.y, v0.Normal.z, 1.0f);
	FLOAT4 normal1 = FLOAT4(v1.Normal.x, v1.Normal.y, v1.Normal.z, 1.0f);
	FLOAT4 normal2 = FLOAT4(v2.Normal.x, v2.Normal.y, v2.Normal.z, 1.0f);

	FLOAT2 texcoord0 = FLOAT2(v0.Texcoord.x, v0.Texcoord.y);
	FLOAT2 texcoord1 = FLOAT2(v1.Texcoord.x, v1.Texcoord.y);
	FLOAT2 texcoord2 = FLOAT2(v2.Texcoord.x, v2.Texcoord.y);

	output[triIndex0] = VertexOutput(position0, normal0, texcoord0);
	output[triIndex1] = VertexOutput(position1, normal1, texcoord1);
	output[triIndex2] = VertexOutput(position2, normal2, texcoord2);

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

void Renderer::DrawTriangles(std::shared_ptr<DeviceBuffer> vertexBuffer, std::shared_ptr<DeviceBuffer> outputBuffer, std::shared_ptr<DeviceBuffer> indexBuffer, unsigned int vertexCount, unsigned int indexCount, const FLOAT4X4& transform, const FLOAT4X4& view, const FLOAT4X4& projection)
{
	unsigned int width = mCanvas->GetWidth();
	unsigned int height = mCanvas->GetHeight();

	void* buffer = mBuffer->GetVirtual();

	dim3 block = dim3(32, 1, 1);
	int left = mTriangleCount % 3;
	dim3 grid = dim3((mTriangleCount / block.x) + left, 1,1);

	SampleVertex* sampleVertices = reinterpret_cast<SampleVertex*>(vertexBuffer->GetVirtual());
	VertexOutput* outputVertices = reinterpret_cast<VertexOutput*>(outputBuffer->GetVirtual());

	unsigned int* indices = reinterpret_cast<unsigned int*>(indexBuffer->GetVirtual());

	KernelTransformVertices<<<grid,block>>>(sampleVertices, outputVertices, indices, vertexCount, mTriangleCount, FLOAT4X4::Identity(), view, projection);
	cudaDeviceSynchronize();
}
