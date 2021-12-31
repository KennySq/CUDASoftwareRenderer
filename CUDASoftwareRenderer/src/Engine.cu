#include<pch.h>
#include"Engine.cuh"
#include"DIB.cuh"
#include"DeviceMemory.cuh"
#include"Color.cuh"

Engine::Engine(HWND hWnd)
	: mHandle(hWnd), mDIB(std::make_shared<DIB>(hWnd, 1280, 720)), 
	mResources(std::make_unique<ResourceManager>()), 
	mRenderer(std::make_unique<Renderer>(mDIB, std::move(mResources)))
{
	mTexture = mResources->CreateTexture2D(1280, 720);

}

void Engine::Start()
{

}

void Engine::Update(float delta)
{
	mRenderer->ClearCanvas(ColorRGBA(1.0f, 1.0f, 0.0f, 0.0f));
}

void Engine::Render(float delta)
{
	mRenderer->Present();
}

void Engine::Destroy()
{
	
}
