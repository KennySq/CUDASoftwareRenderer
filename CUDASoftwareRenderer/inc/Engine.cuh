#pragma once

struct DIB;
struct DeviceMemory;
struct ResourceManager;
struct DeviceTexture;

struct Renderer;

#include"ResourceManager.cuh"
#include"Renderer.cuh"

struct Engine
{
public:
	Engine(HWND hWnd);

	void Start();
	void Update(float delta);
	void Render(float delta);
	void Destroy();
private:
	std::shared_ptr<DIB> mDIB;
	
	std::unique_ptr<ResourceManager> mResources;
	std::unique_ptr<Renderer> mRenderer;

	std::shared_ptr<DeviceTexture> mTexture;



	HWND mHandle;
};