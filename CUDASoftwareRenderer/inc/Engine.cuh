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
	void Update(float delta, float time);
	void Render(float delta);
	void Destroy();

	void RButtonDown(int x, int y);
private:
	std::shared_ptr<DIB> mDIB;
	
	std::shared_ptr<ResourceManager> mResources;
	std::unique_ptr<Renderer> mRenderer;

	std::shared_ptr<DeviceTexture> mTexture;

	std::shared_ptr<DeviceBuffer> mVertexBuffer0;
	std::shared_ptr<DeviceBuffer> mIndexBuffer0;

	std::shared_ptr<DeviceBuffer> mVertexBuffer1;
	std::shared_ptr<DeviceBuffer> mIndexBuffer1;

	std::shared_ptr<DeviceBuffer> mFragmentBuffer0;
	std::shared_ptr<DeviceBuffer> mFragmentBuffer1;
	std::shared_ptr<DeviceBuffer> mTriangleBuffer0;
	std::shared_ptr<DeviceBuffer> mTriangleBuffer1;

	AABB mAABB0;


	unsigned int mVertexCount0;
	unsigned int mIndexCount0;

	unsigned int mVertexCount1;
	unsigned int mIndexCount1;

	HWND mHandle;
};