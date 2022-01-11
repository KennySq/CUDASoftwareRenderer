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

__global__ void KernelClearDepth(void* target, unsigned int width, unsigned int height, float v)
{
	unsigned int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	DWORD* asPixel = reinterpret_cast<DWORD*>(target);

	asPixel[index] = PackDepth(v);
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

	void* texture = mBuffer->GetVirtual();

	dim3 block = dim3(32, 18, 1);
	dim3 grid = dim3(width / block.x, height / block.y, 1);

	KernelClearBitmap << <grid, block >> > (texture, width, height, clearColor);

	cudaDeviceSynchronize();
}

void Renderer::ClearDepth()
{
	unsigned int width = mCanvas->GetWidth();
	unsigned int height = mCanvas->GetHeight();

	void* depth = mDepth->GetVirtual();

	dim3 block = dim3(32, 18, 1);
	dim3 grid = dim3(width / block.x, height / block.y, 1);

	KernelClearDepth << <grid, block >> > (depth, width, height, 0.0f);
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

inline __device__ void DeviceSetPixel(DWORD* buffer, unsigned int pointIndex, DWORD value)
{
	buffer[pointIndex] = value;
}

inline __device__ DWORD DeviceGetPixel(DWORD* buffer, unsigned int pointIndex)
{
	return buffer[pointIndex];
}

__device__ void DeviceGetTriangleDepth(const Renderer::Triangle& triangle, float& d0, float& d1, float& d2)
{
	const float fn0 = (100.0f + 0.01f) / (2.0f * (100.0f - 0.01f));
	const float fn1 = (-100.0f * 0.01f) / (100.0f - 0.01f);

	d0 = fn0 + (1.0f / HomogeneousToNDC(triangle.FragmentInput[0].Position).z) * fn1 + 0.5f;
	d1 = fn0 + (1.0f / HomogeneousToNDC(triangle.FragmentInput[1].Position).z) * fn1 + 0.5f;
	d2 = fn0 + (1.0f / HomogeneousToNDC(triangle.FragmentInput[2].Position).z) * fn1 + 0.5f;
}

__device__ float DeviceGetDepth(DWORD* depth, const INT2& point, unsigned int width)
{
	return depth[PointToIndex(point, width)];
}

__device__ void DeviceGetBarycentricAreas(const INT2& p4, const INT2& p0, const INT2& p1, const INT2& p2, float& u, float& v, float& w)
{
	INT2 ps0 = p1 - p0;
	INT2 ps1 = p2 - p0;
	INT2 ps2 = p4 - p0;

	FLOAT3 v0 = FLOAT3(ps0.x, ps0.y, 0.0f), v1 = FLOAT3(ps1.x, ps1.y, 0.0f), v2 = FLOAT3(ps2.x, ps2.y, 0.0f);

	float d00 = Float3Dot(v0, v0);
	float d01 = Float3Dot(v0, v1);
	float d11 = Float3Dot(v1, v1);
	float d20 = Float3Dot(v2, v0);
	float d21 = Float3Dot(v2, v1);

	float denom = (d00 * d11 - d01 * d01);

	v = (d11 * d20 - d01 * d21) / denom;
	w = (d00 * d21 - d01 * d20) / denom;
	u = 1.0f - v - w;
}

__device__ float DeviceInterpolateDepth(unsigned int width, unsigned int height,
	const Renderer::Triangle& triangle, const INT2& point)
{
	FLOAT4 projected0 = triangle.FragmentInput[0].Position;
	FLOAT4 projected1 = triangle.FragmentInput[1].Position;
	FLOAT4 projected2 = triangle.FragmentInput[2].Position;

	FLOAT3 ndc0 = HomogeneousToNDC(projected0);
	FLOAT3 ndc1 = HomogeneousToNDC(projected1);
	FLOAT3 ndc2 = HomogeneousToNDC(projected2);

	INT2 clip0 = NDCToClipSpace(ndc0, width, height);
	INT2 clip1 = NDCToClipSpace(ndc1, width, height);
	INT2 clip2 = NDCToClipSpace(ndc2, width, height);

	float u, v, w;

	DeviceGetBarycentricAreas(point, clip0, clip1, clip2, u, v, w);

	FLOAT3 barycentric = FLOAT3(u, v, w);

	float z = -(barycentric.x * projected0.z + barycentric.y * projected1.z + barycentric.z * projected2.z);
	
	return 1.0f / z;
}

__device__  void DeviceDrawLine(DWORD* buffer, DWORD* depth, 
	const INT2& p0, const INT2& p1,
	const Renderer::Triangle& triangle, 
	unsigned int width, unsigned int height, const ColorRGBA& debugColor)
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

	INT2 point = INT2(from.x, from.y);

	if (dx > dy)
	{
		for (int i = 0; i <= d; i++)
		{
			unsigned int index = (point.y * width) + point.x;

			float d = DeviceInterpolateDepth(width, height, triangle, point);
			int packed = PackDepth(d);

			atomicMax(reinterpret_cast<int*>(&depth[index]), packed);
			
			if (depth[index] == packed)
			{
				DeviceSetPixel(buffer, index, debugColor);
			}

			point.x += sx;
			r += dy;

			if (r > dx)
			{
				point.y += sy;
				r -= dx;
			}
		}
	}
	else
	{
		for (int i = 0; i <= d; i++)
		{
			unsigned int index = (point.y * width) + point.x;
			float d = DeviceInterpolateDepth(width, height, triangle, point);
			int packed = PackDepth(d);

			atomicMax(reinterpret_cast<int*>(&depth[index]), packed);
			if (depth[index] == packed)
			{
				DeviceSetPixel(buffer, index, debugColor);
			}
			point.y += sy;
			r += dx;
			if (r >= dy)
			{
				point.x += sx;
				r -= dy;
			}

		}
	}
}

__device__ void DeviceFillBottomFlatTriangle(DWORD* buffer, DWORD* depth,
	const INT2& p0, const INT2& p1, const INT2& p2, const Renderer::Triangle& triangle,
	unsigned int width, unsigned int height, unsigned int threadId,
	const ColorRGBA& debugColor)
{
	int p0yOffset = p0.y - threadId;

	float invSlope0 = ((p1.x - p0.x) / (float)(p1.y - p0.y));
	float invSlope1 = ((p2.x - p0.x) / (float)(p2.y - p0.y));

	float curx0 = p0.x - (invSlope0 * threadId);
	float curx1 = p0.x - (invSlope1 * threadId);

	int scanlineOffset = p0.y - p2.y;

	if (threadId > scanlineOffset || p0yOffset < p2.y || p0yOffset < 1)
	{
		return;
	}

	INT2 begin = INT2(curx0, p0yOffset);
	INT2 end = INT2(curx1, p0yOffset);

	DeviceDrawLine(buffer, depth, begin, end, triangle, width, height, debugColor);
}

__device__ void DeviceFillTopFlatTriangle(DWORD* buffer, DWORD* depth,
	const INT2& p0, const INT2& p1, const INT2& p2, const Renderer::Triangle& triangle,
	unsigned int width, unsigned int height, unsigned int threadId,
	const ColorRGBA& debugColor)
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
	DeviceDrawLine(buffer, depth, begin, end, triangle, width, height, debugColor);
}


__device__ void DeviceDrawFilledTriangle(DWORD* buffer, DWORD* depth, const Renderer::Triangle& triangle,
	unsigned int width, unsigned int height, unsigned int threadId)
{
	FLOAT4 project0 = triangle.FragmentInput[0].Position;
	FLOAT4 project1 = triangle.FragmentInput[1].Position;
	FLOAT4 project2 = triangle.FragmentInput[2].Position;

	FLOAT3 ndc0 = HomogeneousToNDC(project0);
	FLOAT3 ndc1 = HomogeneousToNDC(project1);
	FLOAT3 ndc2 = HomogeneousToNDC(project2);

	INT2 cp0 = NDCToClipSpace(ndc0, width, height);
	INT2 cp1 = NDCToClipSpace(ndc1, width, height);
	INT2 cp2 = NDCToClipSpace(ndc2, width, height);

	unsigned int depthIndex0 = PointToIndex(cp0, width);
	unsigned int depthIndex1 = PointToIndex(cp1, width);
	unsigned int depthIndex2 = PointToIndex(cp2, width);

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
		DeviceFillBottomFlatTriangle(buffer, depth, cp0, cp1, cp2, triangle, width, height, threadId, ColorRGBA(1, 1, 1, 1));
	}

	else if (cp0.y == cp1.y)
	{
		DeviceFillTopFlatTriangle(buffer,depth, cp0, cp1, cp2, triangle, width, height, threadId, ColorRGBA(1, 1, 1, 1));
	}
	else
	{
		int midx = (cp0.x + ((float)(cp1.y - cp0.y) / (float)(cp2.y - cp0.y)) * (cp2.x - cp0.x));
		int midy = cp1.y;

		INT2 mid = INT2(midx, midy);

		DeviceFillTopFlatTriangle(buffer, depth, mid, cp1, cp2, triangle, width, height, threadId, ColorRGBA(1, 1, 1, 1));
		DeviceFillBottomFlatTriangle(buffer, depth, cp0, cp1, mid, triangle, width, height, threadId, ColorRGBA(1, 1, 1, 1));
	}
}

__device__ FLOAT3 DeviceGetSurfaceNormal(const FLOAT3& p0, const FLOAT3& p1, const FLOAT3& p2)
{
	FLOAT3 u = p1 - p0;
	FLOAT3 v = p2 - p0;

	return FLOAT3((u.y * v.z) - (u.z * v.y), (u.z * v.x) - (u.x * v.z), (u.x * v.y) - (u.y * v.x));
}

__global__ void KernelRasterize(DWORD* buffer, DWORD* depth, 
	unsigned int width, unsigned int height, 
	Renderer::Triangle* triangles, unsigned int triangleCount, FLOAT3 viewPosition)
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

	FLOAT3 viewDir = triangle.SurfaceNormal - viewPosition;
	if (Float3Dot(Float3Normalize(viewDir), Float3Normalize(triangle.SurfaceNormal)) > 0.0f)
	{
		return;
	}



	int scanlineIndex = threadId % threadPerTriangle;

	DeviceDrawFilledTriangle(buffer, depth, triangle, width, height, scanlineIndex);
}

__device__ AABB DeviceGetAABB(const VertexOutput& vo0, const VertexOutput& vo1, const VertexOutput& vo2, unsigned int width, unsigned int height)
{
	FLOAT3 ndc0 = HomogeneousToNDC(vo0.Position);
	FLOAT3 ndc1 = HomogeneousToNDC(vo1.Position);
	FLOAT3 ndc2 = HomogeneousToNDC(vo2.Position);

	int maxX, maxY;
	int minX, minY;

	INT2 clip0 = NDCToClipSpace(ndc0, width, height);
	INT2 clip1 = NDCToClipSpace(ndc1, width, height);
	INT2 clip2 = NDCToClipSpace(ndc2, width, height);

	maxX = max(clip0.x, clip1.x);
	maxX = max(maxX, clip2.x);

	minX = min(clip0.x, clip1.x);
	minX = min(minX, clip2.x);

	maxY = max(clip0.y, clip1.y);
	maxY = max(maxY, clip2.y);

	minY = min(clip0.y, clip1.y);
	minY = min(minY, clip2.y);

	return AABB(INT2(minX, minY), INT2(maxX, maxY));
}

__device__ FLOAT3 DeviceGetBarycentric(const FLOAT4& p0, const FLOAT4& p1, const FLOAT4& p2)
{
	FLOAT3 ndc0 = HomogeneousToNDC(p0);
	FLOAT3 ndc1 = HomogeneousToNDC(p1);
	FLOAT3 ndc2 = HomogeneousToNDC(p2);

	FLOAT3 q = (ndc1 - ndc2) / 2;

	FLOAT3 p0toq = ndc0 - q;
	FLOAT3 p0top1 = ndc0 - ndc1;
	FLOAT3 p0top2 = ndc0 - ndc2;

	FLOAT3 result;

	float s0 = Float3Dot(p0toq, p0top1);
	float s1 = Float3Dot(p0top2, p0top2);
	float s2 = Float3Dot(p0top1, p0top1);
	float s3 = Float3Dot(p0top1, p0top2);
	float s4 = Float3Dot(p0toq, p0top2);

	result.x = ((s0 * s1) - (s4 * s3)) / ((s2 * s1) - (s3 * s3));
	result.y = ((s4 * s2) - (s0 * s3)) / ((s2 * s1) - (s3 * s3));

	return result;

}

__global__ void KernelTransformVertices(DWORD* buffer, DWORD* depth,
	Renderer::Triangle* triangles,
	unsigned int width, unsigned int height,
	SampleVertex* vertices, VertexOutput* output,
	unsigned int* indices, unsigned int vertexCount,
	unsigned int indexCount, FLOAT4X4 Transform,
	FLOAT4X4 View, FLOAT4X4 Projection)
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

	AABB aabb = DeviceGetAABB(o0, o1, o2, width, height);
	FLOAT3 barycentric = DeviceGetBarycentric(o0.Position, o1.Position, o2.Position);
	FLOAT3 surfaceNormal = DeviceGetSurfaceNormal(v0.Position, v1.Position, v2.Position);
	FLOAT4 wolrdSurfaceNormal = Float4Multiply(FLOAT4(surfaceNormal.x, surfaceNormal.y, surfaceNormal.z, 1.0f), Transform);

	surfaceNormal = FLOAT3(wolrdSurfaceNormal.x, wolrdSurfaceNormal.y, wolrdSurfaceNormal.z);

	triangles[index] = Renderer::Triangle(o0, o1, o2, aabb, barycentric, surfaceNormal);

	// testing project to clip space transform
	FLOAT3 ndcPosition0 = HomogeneousToNDC(position0);
	FLOAT3 ndcPosition1 = HomogeneousToNDC(position1);
	FLOAT3 ndcPosition2 = HomogeneousToNDC(position2);

	INT2 point0 = NDCToClipSpace(ndcPosition0, width, height);
	INT2 point1 = NDCToClipSpace(ndcPosition1, width, height);
	INT2 point2 = NDCToClipSpace(ndcPosition2, width, height);

	DeviceDrawLine(buffer, depth, point0, point1, triangles[index], width, height, ColorRGBA(0, 0, 0, 0));
	DeviceDrawLine(buffer, depth, point1, point2, triangles[index], width, height, ColorRGBA(0, 0, 0, 0));
	DeviceDrawLine(buffer, depth, point2, point0, triangles[index], width, height, ColorRGBA(0, 0, 0, 0));

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

void Renderer::DrawTriangles(std::shared_ptr<DeviceBuffer> vertexBuffer,
	std::shared_ptr<DeviceBuffer> indexBuffer, 
	std::shared_ptr<DeviceBuffer> fragmentBuffer, 
	std::shared_ptr<DeviceBuffer> triangleBuffer, 
	unsigned int vertexCount, unsigned int indexCount, 
	const FLOAT4X4& transform, const FLOAT4X4& view, const FLOAT4X4& projection)
{
	unsigned int width = mCanvas->GetWidth();
	unsigned int height = mCanvas->GetHeight();

	void* buffer = mBuffer->GetVirtual();
	void* depth = mDepth->GetVirtual();

	int totalThread = indexCount / 3;

	dim3 transformBlock = dim3(512, 1, 1);
	dim3 transformGrid = dim3(((totalThread + transformBlock.x - 1) / transformBlock.x), 1, 1);

	dim3 rasterBlock = dim3(28, 28, 1);
	dim3 rasterGrid = dim3((width + rasterBlock.x - 1) / rasterBlock.x, (height + rasterBlock.y - 1) / rasterBlock.y, 1);

	if (transformGrid.x == 0)
	{
		transformGrid.x = 1;
	}

	SampleVertex* sampleVertices = reinterpret_cast<SampleVertex*>(vertexBuffer->GetVirtual());
	VertexOutput* outputVertices = reinterpret_cast<VertexOutput*>(fragmentBuffer->GetVirtual());
	Renderer::Triangle* triangles = reinterpret_cast<Renderer::Triangle*>(triangleBuffer->GetVirtual());

	unsigned int* indices = reinterpret_cast<unsigned int*>(indexBuffer->GetVirtual());

	KernelTransformVertices << <transformGrid, transformBlock >> >
		(CAST_PIXEL(buffer), CAST_PIXEL(depth),
			triangles, width, height,
			sampleVertices, outputVertices,
			indices, vertexCount, indexCount,
			transform, view, projection);

	cudaDeviceSynchronize();
	/*KernelDepth << <rasterGrid, rasterBlock >> > (CAST_PIXEL(depth), width, height, triangles, totalThread, projection);
	cudaDeviceSynchronize();*/

	float determView = Float4x4Determinant(view);

	FLOAT4X4 invView = Float4x4Multiply(view, determView);

	FLOAT3 viewPos = FLOAT3(invView._41, invView._42, invView._43);


	KernelRasterize << <rasterGrid, rasterBlock >> >
		(CAST_PIXEL(buffer), CAST_PIXEL(depth), 
			width, height, triangles, totalThread,
		viewPos);
	cudaDeviceSynchronize();
}
