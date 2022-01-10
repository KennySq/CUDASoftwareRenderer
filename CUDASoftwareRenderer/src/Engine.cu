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
	mFragmentBuffer = mResources->CreateBuffer(sizeof(VertexOutput), mVertexCount);
	mTriangleBuffer = mResources->CreateBuffer(sizeof(Renderer::Triangle), mIndexCount / 3);
}

void Engine::Update(float delta, float time)
{
	mRenderer->ClearCanvas(ColorRGBA(0.0f, 0.0f, 0.0f, 0.0f));
	mRenderer->ClearDepth();
	static FLOAT4X4 transform = Float4x4Multiply(FLOAT4X4::Identity(), Float4x4RotationX(90.0f));
	static FLOAT4X4 view = Float4x4ViewMatrix(0, 0, 0);
	static FLOAT4X4 projection = Float4x4ProjectionMatrix(0.01f, 100.0f, DegreeToRadian(90.0f), 1.777f);
	//mRenderer->Present();

	view._43 = 4.0f;
	/*
	view._42 = -2.0f;
	view._41 = -4.0f;*/
	transform = Float4x4Multiply(transform, Float4x4RotationX(delta));
//	transform = Float4x4Multiply(transform, Float4x4RotationY(delta));
//	transform = Float4x4Multiply(transform, Float4x4RotationZ(delta));

	mRenderer->DrawTriangles(mVertexBuffer, mIndexBuffer, mFragmentBuffer, mTriangleBuffer, mVertexCount, mIndexCount, transform, view, projection);
	mRenderer->OutText(0, 0, std::to_string(1.0f / delta));
}

void Engine::Render(float delta)
{
	mRenderer->Present();

}

void Engine::Destroy()
{
}
