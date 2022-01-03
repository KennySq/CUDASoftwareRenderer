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
	
	std::shared_ptr<ResourceManager> mResources;
	std::unique_ptr<Renderer> mRenderer;

	std::shared_ptr<DeviceTexture> mTexture;

	std::shared_ptr<DeviceBuffer> mVertexBuffer;
	std::shared_ptr<DeviceBuffer> mVertexOutput;

	std::shared_ptr<DeviceBuffer> mIndexBuffer;

	unsigned int mVertexCount;
	unsigned int mIndexCount;

	HWND mHandle;
};