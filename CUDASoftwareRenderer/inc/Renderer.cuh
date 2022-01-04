#pragma once
struct DIB;
struct DeviceTexture;
struct DeviceBuffer;
struct ColorRGBA;
struct ResourceManager;
struct ColorRGBA;

#include"3DMath.cuh"
#include"Color.cuh"

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

	struct Line2D
	{
		__device__ __host__ Line2D()
		{

		}

		__device__ __host__ Line2D(const Point2D& from, const Point2D& to)
			: From(from), To(to)
		{
			if (To.Point < From.Point)
			{
				std::swap(From, To);
			}
		}

		__device__ __host__ Line2D(const Line2D& right)
			: From(right.From), To(right.To)
		{
			if (To.Point < From.Point)
			{
				std::swap(From, To);
			}
		}

		Point2D From, To;
	};

	struct Triangle2D
	{
		__device__ __host__ Triangle2D()
		{

		}

		__device__ __host__ Triangle2D(unsigned int i0, unsigned int i1, unsigned int i2)
			: P0(i0), P1(i1), P2(i2)
		{
		}

		__device__ __host__ Triangle2D(const Triangle2D& right)
			: P0(right.P0), P1(right.P1), P2(right.P2)
		{

		}

		unsigned int P0, P1, P2;
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
	void DrawTriangles(std::shared_ptr<DeviceBuffer> vertexBuffer, std::shared_ptr<DeviceBuffer> outputBuffer, std::shared_ptr<DeviceBuffer> indexBuffer, unsigned int vertexCount, unsigned int indexCount, const FLOAT4X4& transform, const FLOAT4X4& view, const FLOAT4X4& projection);
private:

	std::shared_ptr<DIB> mCanvas;
	std::shared_ptr<DeviceTexture> mBuffer;

	Point2D* mRenderPoints;
	Line2D* mRenderLines;
	std::vector<Triangle2D> mRenderTriangles;

	unsigned int mPointCount;

	unsigned int mTriangleCount;
};