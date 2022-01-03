#pragma once
#include"DeviceEntity.cuh"

struct DeviceMemory;
struct DeviceTexture;
struct DeviceBuffer;
struct ResourceManager
{
public:
	ResourceManager();

	std::shared_ptr<DeviceTexture> CreateTexture2D(unsigned int width, unsigned height);
	std::shared_ptr<DeviceBuffer> CreateBuffer(size_t stride, unsigned int count, void* subResource = nullptr);

private:
	std::vector<DeviceEntity> mResources;
	std::shared_ptr<DeviceMemory> mMemory;
};