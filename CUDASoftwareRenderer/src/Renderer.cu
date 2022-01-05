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

	error = cudaStreamCreate(&mVertexStream);
	CUDAError(error);

	error = cudaStreamCreate(&mFragmentStream);
	CUDAError(error);

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

void Renderer::OutText(int x, int y, std::string str)
{
	TextOutA(mCanvas->GetMemoryDC(), x, y, str.c_str(), str.size());
}

void Renderer::Start()
{

	unsigned int width = mCanvas->GetWidth();
	unsigned int height = mCanvas->GetHeight();

	cudaError_t error = cudaMalloc(reinterpret_cast<void**>(&deviceDrawPoints), width * height * sizeof(Point2D));
	CUDAError(error);


}

void Renderer::Update(float delta)
{

}

void Renderer::Render(float delta)
{
}

void Renderer::Release()
{
	cudaFree(deviceDrawPoints);
	delete[] mRenderPoints;
	mRenderPoints = nullptr;

	cudaStreamDestroy(mVertexStream);
	cudaStreamDestroy(mFragmentStream);
}

void Renderer::ClearCanvas(const ColorRGBA& clearColor)
{
	unsigned int width = mCanvas->GetWidth();
	unsigned int height = mCanvas->GetHeight();

	void* texture = mBuffer->GetVirtual();

	dim3 block = dim3(32, 18, 1);
	dim3 grid = dim3(width / block.x, height / block.y, 1);

	//if (ColorRGBA(0, 0, 0, 0) == clearColor)
	//{
	//	return;
	//}

	KernelClearBitmap << <grid, block >> > (texture, width, height, clearColor);

	cudaDeviceSynchronize();
}

void Renderer::Present()
{
	mCanvas->CopyBuffer(mBuffer);
	mCanvas->Present();
}

inline __device__ void DeviceSetPixel(DWORD* buffer, unsigned int pointIndex, const ColorRGBA& color)
{
	buffer[pointIndex] = ConvertColorToDWORD(color);
}

__device__  void DeviceDrawLine(DWORD* buffer, const INT2& p0, const INT2& p1, unsigned int width, const ColorRGBA& color)
{

	INT2 from = p0;
	INT2 to = p1;

	auto sign = [](int dxy)
	{
		if (dxy < 0)
		{
			return -1;
		}
		else if (dxy > 0)
		{
			return 1;
		}
		else
		{
			return 0;
		}
	};

	int dx = to.x - from.x;
	int dy = to.y - from.y;

	int sx = sign(dx);
	int sy = sign(dy);

	dx = abs(dx);
	dy = abs(dy);

	int d = max(dx, dy);

	double r = d / 2;

	int x = from.x;
	int y = from.y;

	if (dx > dy)
	{
		for (int i = 0; i < d; i++)
		{
			unsigned int index = (y * width) + x;
			DeviceSetPixel(buffer, index, color);

			x += sx;
			r += dy;

			if (r >= dx)
			{
				y += sy;
				r -= dx;
			}
		}
	}
	else
	{
		for (int i = 0; i < d; i++)
		{
			unsigned int index = (y * width) + x;
			DeviceSetPixel(buffer, index, color);

			y += sy;
			r += dx;
			if (r >= dy)
			{
				x += sx;
				r -= dy;
			}

		}
	}


}

__device__ void DeviceFillBottomFlatTriangle(DWORD* buffer, const INT2& p0, const INT2& p1, const INT2& p2, unsigned int width, const ColorRGBA& color)
{
	float invSlope0 = (p1.x - p0.x) / abs(p1.y - p0.y);
	float invSlope1 = (p2.x - p0.x) / abs(p2.y - p0.y);

	float curx0 = p0.x;
	float curx1 = p0.x;

	for (int i = p1.y; i < p0.y; i++)
	{
		INT2 begin = INT2(curx0, i);
		INT2 end = INT2(curx1, i);
		DeviceDrawLine(buffer, begin, end, width, color);

		curx0 += invSlope0;
		curx1 += invSlope1;
	}
}

__device__ void DeviceFillTopFlatTriangle(DWORD* buffer, const INT2& p0, const INT2& p1, const INT2& p2, unsigned int width, const ColorRGBA& color)
{
	float invSlope0 = (p2.x - p0.x) / abs(p2.y - p0.y);
	float invSlope1 = (p2.x - p1.x) / abs(p2.y - p1.y);

	float curx0 = p2.x;
	float curx1 = p2.x;

	for (int i = p0.y; i > p2.y; i--)
	{
		INT2 begin = INT2(curx0, i);
		INT2 end = INT2(curx1, i);
		DeviceDrawLine(buffer, begin, end, width, color);

		curx0 -= invSlope0;
		curx1 -= invSlope1;
	}
}


__device__ void DeviceDrawFilledTriangle(DWORD* buffer, const INT2& p0, const INT2& p1, INT2& p2, unsigned int width, const ColorRGBA& color)
{
	INT2 cp0 = p0, cp1 = p1, cp2 = p2;

	auto sort = [&cp0, &cp1, &cp2]()
	{
		if (cp0.y < cp1.y)
		{
			Swap(cp0, cp1);
		}

		if (cp0.y < cp2.y)
		{
			Swap(cp0, cp2);
		}

		if (cp1.y < cp2.y)
		{
			Swap(cp1, cp2);
		}
	};

	sort();

	if (cp1.y == cp2.y)
	{
		DeviceFillBottomFlatTriangle(buffer, cp0, cp1, cp2, width, color);
	}

	else if (cp0.y == cp1.y)
	{
		DeviceFillTopFlatTriangle(buffer, cp0, cp1, cp2, width, color);
	}
	else
	{
		INT2 mid = INT2((int)(cp0.x + ((float)(cp1.y - cp0.y) / (float)(cp2.y - cp0.y)) * (cp2.x - cp0.x)), cp1.y);

		DeviceFillBottomFlatTriangle(buffer, cp0, cp1, mid, width, color);
		DeviceFillBottomFlatTriangle(buffer, cp1, mid, cp2, width, color);
	}

}

__global__ void KernelRasterize(DWORD* buffer, unsigned int width, unsigned int height, VertexOutput* fragmentInput, unsigned int* indices, unsigned int indexCount)
{


	//FLOAT3 ndcPosition0 = FLOAT3(position0.x / position0.w, position0.y / position0.w, position0.z / position0.w);
	//FLOAT3 ndcPosition1 = FLOAT3(position1.x / position1.w, position1.y / position1.w, position1.z / position1.w);
	//FLOAT3 ndcPosition2 = FLOAT3(position2.x / position2.w, position2.y / position2.w, position2.z / position2.w);

	//INT2 point0 = NDCToScreen(ndcPosition0.x / ndcPosition0.z, ndcPosition0.y / ndcPosition0.z, width, height);
	//INT2 point1 = NDCToScreen(ndcPosition1.x / ndcPosition1.z, ndcPosition1.y / ndcPosition1.z, width, height);
	//INT2 point2 = NDCToScreen(ndcPosition2.x / ndcPosition2.z, ndcPosition2.y / ndcPosition2.z, width, height);

	//clamp(point0);
	//clamp(point1);
	//clamp(point2);

	//DeviceDrawLine(buffer, point0, point1, width, ColorRGBA(1, 0, 0, 0));
	//DeviceDrawLine(buffer, point1, point2, width, ColorRGBA(0, 1, 0, 0));
	//DeviceDrawLine(buffer, point2, point0, width, ColorRGBA(0, 0, 1, 0));
}

__global__ void KernelTransformVertices(DWORD* buffer, unsigned int width, unsigned int height, SampleVertex* vertices, VertexOutput* output, unsigned int* indices, unsigned int vertexCount, unsigned int indexCount, FLOAT4X4 Transform, FLOAT4X4 View, FLOAT4X4 Projection)
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
		+ (threadIdx.z * (blockDim.x * blockDim.y))
		+ (threadIdx.y * blockDim.x) + threadIdx.x;

	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;//blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	unsigned int triThread = index * 3;

	auto clamp = [width, height](INT2& p)
	{
		if (p.x >= width)
		{
			p.x = width - 1;
		}
		else if (p.x < 0)
		{
			p.x = 0;
		}
		if (p.y >= height)
		{
			p.y = height - 1;
		}
		else if (p.y < 0)
		{
			p.y = 0;
		}
	};

	if (triThread + 2 >= indexCount)
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

	VertexOutput o0 = VertexOutput(position0, normal0, texcoord0);
	VertexOutput o1 = VertexOutput(position1, normal1, texcoord1);
	VertexOutput o2 = VertexOutput(position2, normal2, texcoord2);

	output[triIndex0] = o0;
	output[triIndex1] = o1;
	output[triIndex2] = o2;

	FLOAT3 ndcPosition0 = FLOAT3(position0.x / position0.w, position0.y / position0.w, position0.z / position0.w);
	FLOAT3 ndcPosition1 = FLOAT3(position1.x / position1.w, position1.y / position1.w, position1.z / position1.w);
	FLOAT3 ndcPosition2 = FLOAT3(position2.x / position2.w, position2.y / position2.w, position2.z / position2.w);

	INT2 point0 = NDCToScreen(ndcPosition0.x / ndcPosition0.z, ndcPosition0.y / ndcPosition0.z, width, height);
	INT2 point1 = NDCToScreen(ndcPosition1.x / ndcPosition1.z, ndcPosition1.y / ndcPosition1.z, width, height);
	INT2 point2 = NDCToScreen(ndcPosition2.x / ndcPosition2.z, ndcPosition2.y / ndcPosition2.z, width, height);

	clamp(point0);
	clamp(point1);
	clamp(point2);

	DeviceDrawLine(buffer, point0, point1, width, ColorRGBA(1, 0, 0, 0));
	DeviceDrawLine(buffer, point1, point2, width, ColorRGBA(0, 1, 0, 0));
	DeviceDrawLine(buffer, point2, point0, width, ColorRGBA(0, 0, 1, 0));

	//	DeviceDrawFilledTriangle(buffer, point0, point1, point2, width, ColorRGBA(1, 1, 1, 1));
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

void Renderer::DrawTriangles(std::shared_ptr<DeviceBuffer> vertexBuffer, std::shared_ptr<DeviceBuffer> indexBuffer, std::shared_ptr<DeviceBuffer> fragmentBuffer, unsigned int vertexCount, unsigned int indexCount, const FLOAT4X4& transform, const FLOAT4X4& view, const FLOAT4X4& projection)
{
	unsigned int width = mCanvas->GetWidth();
	unsigned int height = mCanvas->GetHeight();

	void* buffer = mBuffer->GetVirtual();

	int totalThread = indexCount / 3;
	dim3 block = dim3(512, 1, 1); // x(triangle) y(width), z(height)
	int left = totalThread % 512;
	dim3 grid = dim3((totalThread / block.x) + left, 1, 1);

	dim3 rasterBlock = dim3(4, 16, 16);
	dim3 rasterGrid = dim3(totalThread / rasterBlock.x + (totalThread % rasterBlock.x), width / rasterBlock.y, height / rasterBlock.z);

	//	dim3 grid = dim3(((indexCount / 3) / block.x) + left, width / block.y, height / block.z);

	if (grid.x == 0)
	{
		grid.x = 1;
	}

	SampleVertex* sampleVertices = reinterpret_cast<SampleVertex*>(vertexBuffer->GetVirtual());
	VertexOutput* outputVertices = reinterpret_cast<VertexOutput*>(fragmentBuffer->GetVirtual());

	unsigned int* indices = reinterpret_cast<unsigned int*>(indexBuffer->GetVirtual());

	KernelTransformVertices << <grid, block >> > (CAST_PIXEL(buffer), width, height, sampleVertices, outputVertices, indices, vertexCount, indexCount, transform, view, projection);
	cudaDeviceSynchronize();

	KernelRasterize << <rasterGrid, rasterBlock >> > (CAST_PIXEL(buffer), width, height, outputVertices, indices, indexCount);
	cudaDeviceSynchronize();
}
