#pragma once
struct DIB;
struct DeviceTexture;
struct DeviceBuffer;
struct ColorRGBA;
struct ResourceManager;
struct ColorRGBA;

#include"3DMath.cuh"
#include"Color.cuh"
#include"Geometry.cuh"
#include"DeviceMemory.cuh"
#include"DeviceVector.cuh"

#define MAX_TILE_CAPACITY 2048

struct Renderer
{
public:

	struct Point2D
	{
		__device__ __host__ Point2D()
			: Point(0,0), Color(0,0,0,0)
		{

		}

		__device__ __host__ Point2D(const INT2& point, const ColorRGBA& color)
			: Point(point), Color(color) {}

		__device__ __host__ Point2D(const Point2D& right)
			: Point(right.Point), Color(right.Color)
		{

		}
		INT2 Point;
		ColorRGBA Color;
	};

	struct Triangle
	{
		__device__ __host__ Triangle()
		{

		}

		__device__ __host__ Triangle(const VertexOutput& vo0, const VertexOutput& vo1, const VertexOutput& vo2, const AABB2D& aabb, const FLOAT3& barycentric, const FLOAT3& surfaceNormal)
			: Bound(aabb), Barycentric(barycentric), SurfaceNormal(surfaceNormal)
		{
			FragmentInput[0] = vo0;
			FragmentInput[1] = vo1;
			FragmentInput[2] = vo2;
		}

		__device__ __host__ Triangle(const Triangle& right)
			: Bound(right.Bound), Barycentric(right.Barycentric), SurfaceNormal(right.SurfaceNormal)
		{
			FragmentInput[0] = right.FragmentInput[0];
			FragmentInput[1] = right.FragmentInput[1];
			FragmentInput[2] = right.FragmentInput[2];
		}

		VertexOutput FragmentInput[3];
		AABB2D Bound;
		FLOAT3 Barycentric;
		FLOAT3 SurfaceNormal;
	};
	
	struct Tile
	{
		__device__ Tile()
			: Count(0)
		{

		}

		__device__ ~Tile()
		{

		}

		Renderer::Triangle* Triangles[MAX_TILE_CAPACITY];
		unsigned int Count;
	};


	Renderer(std::shared_ptr<DIB> dib, std::shared_ptr<ResourceManager> rs);
	~Renderer();

	void Start();
	void Update(float delta);
	void Render(float delta);
	void Release();

	void ClearCanvas(const ColorRGBA& clearColor);
	void ClearDepth();
	void DrawTexture(std::shared_ptr<DeviceTexture> texture, int x, int y);

	void OutText(int x, int y, std::string str);

	void Present();
	
	void DrawTriangles(std::shared_ptr<DeviceBuffer> vertexBuffer, std::shared_ptr<DeviceBuffer> indexBuffer,
		std::shared_ptr<DeviceBuffer> fragmentBuffer, std::shared_ptr<DeviceBuffer> triangleBuffer, 
		unsigned int vertexCount, unsigned int indexCount, 
		const FLOAT4X4& transform, const FLOAT4X4& view, const FLOAT4X4& projection);

	void BindTexture(std::shared_ptr<DeviceTexture> texture, unsigned int index);
	
private:

	std::shared_ptr<DIB> mCanvas;
	std::shared_ptr<DeviceTexture> mBuffer;
	std::shared_ptr<DeviceTexture> mDepth;

	cudaStream_t mVertexStream;
	cudaStream_t mFragmentStream;

	Point2D* mRenderPoints;

	dim3 mRasterizerGrid;
	dim3 mRasterizerBlock;
	std::shared_ptr<DeviceBuffer> mTileBuffer;

	unsigned int mPointCount;

	unsigned int mTriangleCount;
};