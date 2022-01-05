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
	void DrawTriangles(std::shared_ptr<DeviceBuffer> vertexBuffer, std::shared_ptr<DeviceBuffer> indexBuffer, std::shared_ptr<DeviceBuffer> fragmentBuffer, unsigned int vertexCount, unsigned int indexCount, const FLOAT4X4& transform, const FLOAT4X4& view, const FLOAT4X4& projection);

	
private:

	std::shared_ptr<DIB> mCanvas;
	std::shared_ptr<DeviceTexture> mBuffer;

	cudaStream_t mVertexStream;
	cudaStream_t mFragmentStream;

	Point2D* mRenderPoints;

	unsigned int mPointCount;

	unsigned int mTriangleCount;
};