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

	std::string resourcePath0 = workingPath;
	std::string resourcePath1 = workingPath;

	resourcePath0 += "\\CUDASoftwareRenderer\\assets\\steve.fbx";
	resourcePath1 += "\\CUDASoftwareRenderer\\assets\\sphere.fbx";

	FbxLoader sampleLoader0(resourcePath0.c_str());
	FbxLoader sampleLoader1(resourcePath1.c_str());

	mVertexCount0 = sampleLoader0.Vertices.size();
	mIndexCount0 = sampleLoader0.Indices.size();

	mVertexCount1 = sampleLoader1.Vertices.size();
	mIndexCount1 = sampleLoader1.Indices.size();


	std::vector<SampleVertex> vertices0;
	std::vector<SampleVertex> vertices1;

	FLOAT3 maxPosition0 = FLOAT3(FLT_MIN, FLT_MIN, FLT_MIN);
	FLOAT3 minPosition0 = FLOAT3(FLT_MAX, FLT_MAX, FLT_MAX);

	for (unsigned int i = 0; i < mVertexCount0; i++)
	{
		Vertex conv = sampleLoader0.Vertices[i];
		
		float dist = conv.mPosition.x + conv.mPosition.y + conv.mPosition.z;
		float compMax = maxPosition0.x + maxPosition0.y + maxPosition0.z;
		float compMin = minPosition0.x + minPosition0.y + minPosition0.z;

		if (dist > compMax)
		{
			maxPosition0 = FLOAT3(conv.mPosition.x, conv.mPosition.y, conv.mPosition.z);
		}

		if (dist < compMin)
		{
			minPosition0 = FLOAT3(conv.mPosition.x, conv.mPosition.y, conv.mPosition.z);
		}

		vertices0.push_back(ConvertVertex(conv));
	}

	mAABB0 = AABB(minPosition0, maxPosition0);

	for (unsigned int i = 0; i < mVertexCount1; i++)
	{
		Vertex conv = sampleLoader1.Vertices[i];
		vertices1.push_back(ConvertVertex(conv));
	}

	mVertexBuffer0 = mResources->CreateBuffer(sizeof(SampleVertex), mVertexCount0, vertices0.data());
	mIndexBuffer0 = mResources->CreateBuffer(sizeof(unsigned int), mIndexCount0, sampleLoader0.Indices.data());
	
	mVertexBuffer1 = mResources->CreateBuffer(sizeof(SampleVertex), mVertexCount1, vertices1.data());
	mIndexBuffer1 = mResources->CreateBuffer(sizeof(unsigned int), mIndexCount1, sampleLoader1.Indices.data());

	
	mFragmentBuffer0 = mResources->CreateBuffer(sizeof(VertexOutput), mVertexCount0);
	mFragmentBuffer1 = mResources->CreateBuffer(sizeof(VertexOutput), mVertexCount1);

	mTriangleBuffer0 = mResources->CreateBuffer(sizeof(Renderer::Triangle), mIndexCount0 / 3);
	mTriangleBuffer1 = mResources->CreateBuffer(sizeof(Renderer::Triangle), mIndexCount1 / 3);

	DWORD packedDepth = PackDepth(0.998f);
}

void Engine::Update(float delta, float time)
{
	mRenderer->ClearCanvas(ColorRGBA(0.0f, 0.0f, 0.0f, 0.0f));
	mRenderer->ClearDepth();
	static FLOAT4X4 transform0 = Float4x4Multiply(FLOAT4X4::Identity(), Float4x4RotationX(-90.0f));
	static FLOAT4X4 transform1 = Float4x4Multiply(Float4x4Multiply(Float4x4Translate(FLOAT3(-3, 0, 0) ), Float4x4RotationX(-90.0f)), Float4x4Scale(FLOAT3(10, 10, 10)));
	static FLOAT4X4 view = Float4x4ViewMatrix(0, 0, 0);
	static FLOAT4X4 projection = Float4x4ProjectionMatrix(0.01f, 100.0f, DegreeToRadian(90.0f), 1.777f);

	view._42 = -1.0f;
	view._43 = 50.0f;// +(sin(time) * 20.0f);
	view._41 = (sin(time) * 80.0f);


	//transform = Float4x4Multiply(transform, Float4x4RotationX(delta));
	transform0 = Float4x4Multiply(transform0, Float4x4RotationY(delta));
	//transform = Float4x4Multiply(transform, Float4x4RotationZ(delta));
	Frustum viewFrustum = GetFrustum(Float4x4Multiply(Float4x4Multiply(transform0, view), projection));

	int cullResult = AABBFrustum(mAABB0, viewFrustum);

	if (cullResult >= 0)
	{
		mRenderer->DrawTriangles(mVertexBuffer0, mIndexBuffer0, mFragmentBuffer0, mTriangleBuffer0, mVertexCount0, mIndexCount0, transform0, view, projection);
		printf("Not Culled\n");
	}
	else
	{
		printf("Culled\n");
	}

	//mRenderer->DrawTriangles(mVertexBuffer1, mIndexBuffer1, mFragmentBuffer1, mTriangleBuffer1, mVertexCount1, mIndexCount1, transform1, view, projection);
	mRenderer->OutText(0, 0, std::to_string(1.0f / delta));

	
}

void Engine::Render(float delta)
{
	mRenderer->Present();

}

void Engine::Destroy()
{
}

void Engine::RButtonDown(int x, int y)
{
	std::string coord = std::string("(") + std::to_string(x) + ", " + std::to_string(y) + ")";
	mRenderer->OutText(x, y, coord);
}
