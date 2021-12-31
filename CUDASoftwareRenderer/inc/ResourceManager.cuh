#pragma once
#include"DeviceEntity.cuh"

struct DeviceMemory;
struct DeviceTexture;
struct ResourceManager
{
public:
	ResourceManager();

	std::shared_ptr<DeviceTexture> CreateTexture2D(unsigned int width, unsigned height);

private:
	std::vector<DeviceEntity> mResources;
	std::shared_ptr<DeviceMemory> mMemory;
};