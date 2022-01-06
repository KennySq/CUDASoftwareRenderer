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

		__device__ __host__ Triangle(const VertexOutput& vo0, const VertexOutput& vo1, const VertexOutput& vo2, const AABB& aabb)
			: Bound(aabb)
		{
			FragmentInput[0] = vo0;
			FragmentInput[1] = vo1;
			FragmentInput[2] = vo2;
		}

		__device__ __host__ Triangle(const Triangle& right)
		{
			FragmentInput[0] = right.FragmentInput[0];
			FragmentInput[1] = right.FragmentInput[1];
			FragmentInput[2] = right.FragmentInput[2];
		}

		VertexOutput FragmentInput[3];
		AABB Bound;
		FLOAT3 Barycentric;

	};
	
	Renderer(std::shared_ptr<DIB> dib, std::shared_ptr<ResourceManager> rs);
	~Renderer();

	void Start();
	void Update(float delta);
	void Render(float delta);
	void Release();

	void ClearCanvas(const ColorRGBA& clearColor);

	void SetPixel(int x, int y, const ColorRGBA& color);
	void SetPixelNDC(float x, float y, const ColorRGBA& color);
	void SetTriangle(const Point2D& p0, const Point2D& p1, const Point2D& p2);

	void OutText(int x, int y, std::string str);

	void Present();
	
	void DrawScreen();
	void DrawTriangles(std::shared_ptr<DeviceBuffer> vertexBuffer, std::shared_ptr<DeviceBuffer> indexBuffer,
		std::shared_ptr<DeviceBuffer> fragmentBuffer, std::shared_ptr<DeviceBuffer> triangleBuffer, 
		unsigned int vertexCount, unsigned int indexCount, 
		const FLOAT4X4& transform, const FLOAT4X4& view, const FLOAT4X4& projection);

	
private:

	std::shared_ptr<DIB> mCanvas;
	std::shared_ptr<DeviceTexture> mBuffer;

	cudaStream_t mVertexStream;
	cudaStream_t mFragmentStream;

	Point2D* mRenderPoints;

	unsigned int mPointCount;

	unsigned int mTriangleCount;
};