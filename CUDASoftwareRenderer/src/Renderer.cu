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
	mDepth = rs->CreateTexture2D(width, height);

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
	RECT rect{};
	rect.right = x;
	rect.bottom = y;

	DrawTextA(mCanvas->GetHandleDC(), str.c_str(), str.size(), &rect, DT_BOTTOM | DT_INTERNAL | DT_NOCLIP);
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

	void* texture = mDepth->GetVirtual();

	dim3 block = dim3(32, 18, 1);
	dim3 grid = dim3(width / block.x, height / block.y, 1);

	KernelClearBitmap << <grid, block >> > (texture, width, height, clearColor);

	cudaDeviceSynchronize();
}

void Renderer::Present()
{
	mCanvas->CopyBuffer(mDepth);
	mCanvas->Present();
}

inline __device__ void DeviceSetPixel(DWORD* buffer, unsigned int pointIndex, const ColorRGBA& color)
{
	buffer[pointIndex] = ConvertColorToDWORD(color);
}

inline __device__ void DeviceSetPixel(DWORD* buffer, unsigned int pointIndex, DWORD value)
{
	buffer[pointIndex] = value;
}

inline __device__ DWORD DeviceGetPixel(DWORD* buffer, unsigned int pointIndex)
{
	return buffer[pointIndex];
}

__device__  void DeviceDrawLine(DWORD* buffer, const INT2& p0, const INT2& p1, unsigned int width, unsigned int height, const ColorRGBA& color)
{
	INT2 from = p0;
	INT2 to = p1;

	Clamp<int>(from.x, 0, width - 1);
	Clamp<int>(from.y, 0, height - 1);

	Clamp<int>(to.x, 0, width - 1);
	Clamp<int>(to.y, 0, height - 1);

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

	double r = (double)d / 2.0f;

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

__device__ void DeviceFillBottomFlatTriangle(DWORD* buffer, 
	const INT2& p0, const INT2& p1, const INT2& p2,
	float d0, float d1, float d2,
	unsigned int width, unsigned int height, const ColorRGBA& color, unsigned int threadId)
{
	int p0yOffset = p0.y - threadId;

	float invSlope0 = (p1.x - p0.x) / floor((float)(p1.y - p0.y));
	float invSlope1 = (p2.x - p0.x) / floor((float)(p2.y - p0.y));

	float curx0 = p0.x - (invSlope0 * (float)threadId);
	float curx1 = p0.x - (invSlope1 * (float)threadId);

	unsigned int scanlineOffset = p0.y - p2.y;

	if (threadId > scanlineOffset || p0yOffset < p2.y || p0yOffset < 0)
	{
		return;
	}

	INT2 begin = INT2(curx0, p0yOffset);
	INT2 end = INT2(curx1, p0yOffset);
	DeviceDrawLine(buffer, begin, end, width, height, color);
}

__device__ void DeviceFillTopFlatTriangle(DWORD* buffer,
	const INT2& p0, const INT2& p1, const INT2& p2,
	float d0, float d1, float d2,
	unsigned int width, unsigned int height, 
	const ColorRGBA& color, unsigned int threadId)
{
	int p2yOffset = p2.y + threadId;

	float invSlope0 = ((p2.x - p0.x) / (float)(p2.y - p0.y));
	float invSlope1 = ((p2.x - p1.x) / (float)(p2.y - p1.y));

	float curx0 = p2.x + (invSlope0 * threadId);
	float curx1 = p2.x + (invSlope1 * threadId);

	int scanlineSize = p0.y - p2.y;

	if (threadId > scanlineSize || p2yOffset > p0.y || p2yOffset < 0)
	{
		return;
	}

	INT2 begin = INT2(curx0, p2yOffset);
	INT2 end = INT2(curx1, p2yOffset);
	DeviceDrawLine(buffer, begin, end, width, height, color);
}


__device__ void DeviceDrawFilledTriangle(DWORD* buffer,
	const INT2& p0, const INT2& p1, INT2& p2, float d0, float d1, float d2,
	unsigned int width, unsigned int height, 
	const ColorRGBA& color, unsigned int threadId)
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

	if (cp2.y == cp1.y)
	{
		DeviceFillBottomFlatTriangle(buffer, cp0, cp1, cp2, width,height, color, threadId);
	}

	else if (cp0.y == cp1.y)
	{
		DeviceFillTopFlatTriangle(buffer, cp0, cp1, cp2, width, height, color, threadId);
	}
	else
	{
		int midx = (cp0.x + ((float)(cp1.y - cp0.y) / (float)(cp2.y - cp0.y)) * (cp2.x - cp0.x));
		int midy = cp1.y;
		
		INT2 mid = INT2(midx, midy);

		DeviceFillTopFlatTriangle(buffer, mid, cp1, cp2, width, height, color, threadId);
		DeviceFillBottomFlatTriangle(buffer, cp0, cp1, mid, width, height, color, threadId);
	}

}

__global__ void KernelRasterize(DWORD* buffer, DWORD* depth, unsigned int width, unsigned int height, Renderer::Triangle* triangles, unsigned int triangleCount, const FLOAT4X4& projection)
{
	unsigned int dispatchThreads = gridDim.y * gridDim.x * blockDim.x * blockDim.y;
	unsigned int threadPerTriangle = dispatchThreads / triangleCount;

	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y)
		+ (threadIdx.y * blockDim.x) + threadIdx.x;

	unsigned int triIndex = (threadId / threadPerTriangle);

	if (triIndex >= triangleCount)
	{
		return;
	}

	Renderer::Triangle triangle = triangles[triIndex];

	FLOAT4 project0 = triangle.FragmentInput[0].Position;
	FLOAT4 project1 = triangle.FragmentInput[1].Position;
	FLOAT4 project2 = triangle.FragmentInput[2].Position;


	FLOAT3 ndc0 = HomogeneousToNDC(project0);
	FLOAT3 ndc1 = HomogeneousToNDC(project1);
	FLOAT3 ndc2 = HomogeneousToNDC(project2);

	INT2 c0 = NDCToClipSpace(ndc0, width, height);
	INT2 c1 = NDCToClipSpace(ndc1, width, height);
	INT2 c2 = NDCToClipSpace(ndc2, width, height);

	unsigned int depthIndex0 = PointToIndex(c0, width);
	unsigned int depthIndex1 = PointToIndex(c1, width);
	unsigned int depthIndex2 = PointToIndex(c2, width);

	float d0 = depth[depthIndex0];
	float d1 = depth[depthIndex1];
	float d2 = depth[depthIndex2];

	int scanlineIndex = threadId % threadPerTriangle;

	DeviceDrawFilledTriangle(buffer, c0, c1, c2, d0,d1,d2, width, height, ColorRGBA(1, 1, 1, 1), scanlineIndex);
}

__device__ AABB DeviceGetAABB(const VertexOutput& vo0, const VertexOutput& vo1, const VertexOutput& vo2, unsigned int width, unsigned int height)
{
	FLOAT3 ndc0 = HomogeneousToNDC(vo0.Position);
	FLOAT3 ndc1 = HomogeneousToNDC(vo1.Position);
	FLOAT3 ndc2 = HomogeneousToNDC(vo2.Position);

	float maxX, maxY;
	float minX, minY;

	ndc0.x *= width;
	ndc0.y *= height;

	ndc1.x *= width;
	ndc1.y *= height;

	ndc2.x *= width;
	ndc2.y *= height;

	maxX = max(ndc0.x, ndc1.x);
	maxX = max(maxX, ndc2.x);

	minX = min(ndc0.x, ndc1.x);
	minX = min(minX, ndc2.x);

	maxY = max(ndc0.y, ndc1.y);
	maxY = max(maxY, ndc2.y);

	minY = min(ndc0.y, ndc1.y);
	minY = min(minY, ndc2.y);

	return AABB(FLOAT2(minX, minY), FLOAT2(maxX, maxY));
}

__global__ void KernelInputAssembly(unsigned int triangleCount, VertexOutput* vertexOutput, Renderer::Triangle* triangleBuffer, unsigned int width, unsigned int height)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int triThread = index * 3;

	VertexOutput vo0 = vertexOutput[triThread];
	VertexOutput vo1 = vertexOutput[triThread + 1];
	VertexOutput vo2 = vertexOutput[triThread + 2];

	AABB aabb = DeviceGetAABB(vo0, vo1, vo2, width, height);

	if (index >= triangleCount)
	{
		return;
	}

	triangleBuffer[index] = Renderer::Triangle(vo0, vo1, vo2, aabb);

	return;
}

__global__ void KernelDepth(DWORD* depth, unsigned int width, unsigned int height, Renderer::Triangle* triangles, unsigned int triangleCount, FLOAT4X4 projection)
{
	unsigned int dispatchThreads = gridDim.y * gridDim.x * blockDim.x * blockDim.y;
	unsigned int threadPerTriangle = dispatchThreads / triangleCount;

	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y)
		+ (threadIdx.y * blockDim.x) + threadIdx.x;

	unsigned int triIndex = (threadId / threadPerTriangle);

	if (triIndex >= triangleCount)
	{
		return;
	}

	Renderer::Triangle triangle = triangles[triIndex];

	FLOAT4 project0 = triangle.FragmentInput[0].Position;
	FLOAT4 project1 = triangle.FragmentInput[1].Position;
	FLOAT4 project2 = triangle.FragmentInput[2].Position;

	const float fn0 = projection._33;
	const float fn1 = projection._34;

	float depth0 = fn0 + (1.0f / project0.z) * fn1;
	float depth1 = fn0 + (1.0f / project1.z) * fn1;
	float depth2 = fn0 + (1.0f / project2.z) * fn1;

	FLOAT3 ndc0 = HomogeneousToNDC(project0);
	FLOAT3 ndc1 = HomogeneousToNDC(project1);
	FLOAT3 ndc2 = HomogeneousToNDC(project2);

	INT2 c0 = NDCToClipSpace(ndc0, width, height);
	INT2 c1 = NDCToClipSpace(ndc1, width, height);
	INT2 c2 = NDCToClipSpace(ndc2, width, height);

	unsigned int index0 = (c0.y * width) + c0.x;
	unsigned int index1 = (c1.y * width) + c1.x;
	unsigned int index2 = (c2.y * width) + c2.x;

	DeviceSetPixel(depth, index0, PackDepth(depth0));
	DeviceSetPixel(depth, index1, PackDepth(depth1));
	DeviceSetPixel(depth, index2, PackDepth(depth2));
}

__global__ void KernelTransformVertices(DWORD* buffer, DWORD* depth, unsigned int width, unsigned int height, SampleVertex* vertices, VertexOutput* output, unsigned int* indices, unsigned int vertexCount, unsigned int indexCount, FLOAT4X4 Transform, FLOAT4X4 View, FLOAT4X4 Projection)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;//blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	unsigned int triThread = index * 3;

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
//33, 34

	output[triThread] = o0;
	output[triThread + 1] = o1;
	output[triThread + 2] = o2;

	//return;

	// testing project to clip space transform
	FLOAT3 ndcPosition0 = HomogeneousToNDC(position0);
	FLOAT3 ndcPosition1 = HomogeneousToNDC(position1);
	FLOAT3 ndcPosition2 = HomogeneousToNDC(position2);

	INT2 point0 = NDCToClipSpace(ndcPosition0, width, height);
	INT2 point1 = NDCToClipSpace(ndcPosition1, width, height);
	INT2 point2 = NDCToClipSpace(ndcPosition2, width, height);


	//DeviceDrawLine(buffer, point0, point1, width, height, ColorRGBA(1, 0, 0, 0));

	//DeviceDrawLine(buffer, point1, point2, width, height, ColorRGBA(0, 1, 0, 0));

	//DeviceDrawLine(buffer, point2, point0, width, height, ColorRGBA(0, 0, 1, 0));

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

	KernelDrawCallSetPixel << <grid, block >> > (CAST_PIXEL(buffer), deviceDrawPoints, width * height, width);

	cudaDeviceSynchronize();
}

void Renderer::DrawTriangles(std::shared_ptr<DeviceBuffer> vertexBuffer, std::shared_ptr<DeviceBuffer> indexBuffer, std::shared_ptr<DeviceBuffer> fragmentBuffer, std::shared_ptr<DeviceBuffer> triangleBuffer, unsigned int vertexCount, unsigned int indexCount, const FLOAT4X4& transform, const FLOAT4X4& view, const FLOAT4X4& projection)
{
	unsigned int width = mCanvas->GetWidth();
	unsigned int height = mCanvas->GetHeight();

	void* buffer = mBuffer->GetVirtual();
	void* depth = mDepth->GetVirtual();

	int totalThread = indexCount / 3;

	dim3 transformBlock = dim3(512, 1, 1);
	dim3 transformGrid = dim3(((totalThread + transformBlock.x - 1) / transformBlock.x), 1, 1);

	dim3 rasterBlock = dim3(32, 32, 1);
	dim3 rasterGrid = dim3((width + rasterBlock.x - 1) / rasterBlock.x, (height + rasterBlock.y - 1) / rasterBlock.y, 1);

	if (transformGrid.x == 0)
	{
		transformGrid.x = 1;
	}

	SampleVertex* sampleVertices = reinterpret_cast<SampleVertex*>(vertexBuffer->GetVirtual());
	VertexOutput* outputVertices = reinterpret_cast<VertexOutput*>(fragmentBuffer->GetVirtual());
	Renderer::Triangle* triangles = reinterpret_cast<Renderer::Triangle*>(triangleBuffer->GetVirtual());

	unsigned int* indices = reinterpret_cast<unsigned int*>(indexBuffer->GetVirtual());

	KernelTransformVertices << <transformGrid, transformBlock >> > (CAST_PIXEL(buffer), CAST_PIXEL(depth), width, height, sampleVertices, outputVertices, indices, vertexCount, indexCount, transform, view, projection);
	cudaDeviceSynchronize();

	KernelInputAssembly << <transformGrid, transformBlock >> > (totalThread, outputVertices, triangles, width, height);
	cudaDeviceSynchronize();

	KernelDepth << <rasterGrid, rasterBlock >> > (CAST_PIXEL(depth), width, height, triangles, indexCount / 3, projection);

	KernelRasterize << <rasterGrid, rasterBlock >> >
		(CAST_PIXEL(buffer), CAST_PIXEL(depth), width, height, triangles, totalThread, projection);
	cudaDeviceSynchronize();
}
