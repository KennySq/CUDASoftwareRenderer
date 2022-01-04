#include<pch.h>
#include"Engine.cuh"
#include"DIB.cuh"
#include"DeviceMemory.cuh"
#include"Color.cuh"
#include"Geometry.cuh"

Engine::Engine(HWND hWnd)
	: mHandle(hWnd), mDIB(std::make_shared<DIB>(hWnd, 1280, 720)),
	mResources(ResourceManager::GetInstance()),
	mRenderer(std::make_unique<Renderer>(mDIB, mResources))
{
	mTexture = mResources->CreateTexture2D(1280, 720);

}

void Engine::Start()
{
	char buffer[256];
	GetModuleFileNameA(nullptr, buffer, 256);
	std::string workingPath = buffer;

	workingPath = workingPath.substr(0, workingPath.find_last_of("\\"));
	workingPath = workingPath.substr(0, workingPath.find_last_of("\\"));
	workingPath = workingPath.substr(0, workingPath.find_last_of("\\"));

	workingPath += "\\CUDASoftwareRenderer\\assets\\cube.fbx";

	FbxLoader sampleLoader(workingPath.c_str());

	mVertexCount = sampleLoader.Vertices.size();
	mIndexCount = sampleLoader.Indices.size();
	
	std::vector<SampleVertex> vertices;

	for (unsigned int i = 0; i < mVertexCount; i++)
	{
		Vertex conv = sampleLoader.Vertices[i];
		vertices.push_back(ConvertVertex(conv));
	}

	mVertexBuffer = mResources->CreateBuffer(sizeof(SampleVertex), mVertexCount, vertices.data());
	mIndexBuffer = mResources->CreateBuffer(sizeof(unsigned int), mIndexCount, sampleLoader.Indices.data());

	mVertexOutput = mResources->CreateBuffer(sizeof(VertexOutput), mVertexCount);

	Renderer::Point2D p0 = Renderer::Point2D(INT2(0, 0), ColorRGBA(1.0f, 0.0f, 0.0f, 0.0f));
	Renderer::Point2D p1 = Renderer::Point2D(INT2(50, 0), ColorRGBA(1.0f, 0.0f, 0.0f, 0.0f));
	Renderer::Point2D p2 = Renderer::Point2D(INT2(25, 50), ColorRGBA(1.0f, 0.0f, 0.0f, 0.0f));



	// Math Test
	FLOAT4X4 View = Float4x4ViewMatrix(0, 0, 0);
	View._43 = -5.0f;
	View._42 = -0.5f;
	FLOAT4X4 Projection = Float4x4ProjectionMatrix(0.01f, 100.0f, DegreeToRadian(90.0f), 1.777f);

	//FLOAT4 tp0 = FLOAT4(0.0f, 1.0f, -1.0f, 1.0f);
	//FLOAT4 tp1 = FLOAT4(0.5f, 0.0f, -1.0f, 1.0f);
	//FLOAT4 tp2 = FLOAT4(-0.5f, 0.0f, -1.0f, 1.0f);

	//// update from here 22/01/03 2:52 PM
	//tp0 = Float4Multiply(tp0, View);
	//tp1 = Float4Multiply(tp1, View);
	//tp2 = Float4Multiply(tp2, View);

	//tp0 = Float4Multiply(tp0, Projection);
	//tp1 = Float4Multiply(tp1, Projection);
	//tp2 = Float4Multiply(tp2, Projection);

	//mRenderer->SetPixelNDC(tp0.x / tp0.z, tp0.y / tp0.z, ColorRGBA(1, 1, 1, 1));
	//mRenderer->SetPixelNDC(tp1.x / tp1.z, tp1.y / tp1.z, ColorRGBA(1, 0, 0, 1));
	//mRenderer->SetPixelNDC(tp2.x / tp2.z, tp2.y / tp2.z, ColorRGBA(0, 1, 0, 1));

}

void Engine::Update(float delta)
{
	mRenderer->ClearCanvas(ColorRGBA(0.0f, 0.0f, 0.0f, 0.0f));

	static FLOAT4X4 transform = FLOAT4X4::Identity();
	static FLOAT4X4 view = Float4x4ViewMatrix(0, 0, 0);
	static FLOAT4X4 projection = Float4x4ProjectionMatrix(0.01f, 100.0f, DegreeToRadian(90.0f), 1.777f);
	//mRenderer->Present();

	view._43 = -8.0f;
	view._41 = 3.0f;
	view._42 = 3.0f;

	transform = Float4x4Multiply(transform, Float4x4RotationX(0.0016f));
	//transform = Float4x4Transpose(transform);

	mRenderer->DrawTriangles(mVertexBuffer, mVertexOutput, mIndexBuffer, mVertexCount, mIndexCount, transform, view, Float4x4Transpose(projection));
	mRenderer->OutText(0, 0, std::to_string(delta));


}

void Engine::Render(float delta)
{
	mRenderer->Present();

}

void Engine::Destroy()
{
}
