#include<pch.h>
#include"Engine.cuh"
#include"DIB.cuh"
#include"DeviceMemory.cuh"
#include"Color.cuh"

Engine::Engine(HWND hWnd)
	: mHandle(hWnd), mDIB(std::make_shared<DIB>(hWnd, 1280, 720)),
	mResources(std::make_shared<ResourceManager>()),
	mRenderer(std::make_unique<Renderer>(mDIB, mResources))
{
	mTexture = mResources->CreateTexture2D(1280, 720);

}

void Engine::Start()
{
	Renderer::Point2D p0 = Renderer::Point2D(INT2(0, 0), ColorRGBA(1.0f, 0.0f, 0.0f, 0.0f));
	Renderer::Point2D p1 = Renderer::Point2D(INT2(50, 0), ColorRGBA(1.0f, 0.0f, 0.0f, 0.0f));
	Renderer::Point2D p2 = Renderer::Point2D(INT2(25, 50), ColorRGBA(1.0f, 0.0f, 0.0f, 0.0f));

	mRenderer->SetTriangle(p0, p1, p2);
}

void Engine::Update(float delta)
{
	mRenderer->ClearCanvas(ColorRGBA(0.0f, 0.0f, 0.0f, 0.0f));


}

void Engine::Render(float delta)
{

	mRenderer->Present();

}

void Engine::Destroy()
{
}
