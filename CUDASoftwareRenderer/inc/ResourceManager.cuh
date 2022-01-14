#pragma once
#include"DeviceEntity.cuh"

struct DeviceMemory;
struct DeviceTexture;
struct DeviceBuffer;
struct ResourceManager
{
public:
	ResourceManager();

	static std::shared_ptr<ResourceManager> GetInstance()
	{
		if (mInstance == nullptr)
		{
			mInstance = std::make_shared<ResourceManager>();
		}

		return mInstance;
	}

	std::shared_ptr<DeviceTexture> CreateTextureFromFile(const char* path);
	std::shared_ptr<DeviceTexture> CreateTexture2D(unsigned int width, unsigned height);
	std::shared_ptr<DeviceBuffer> CreateBuffer(size_t stride, unsigned int count, void* subResource = nullptr);

private:
	static std::shared_ptr<ResourceManager> mInstance;

	std::vector<DeviceEntity> mResources;
	std::shared_ptr<DeviceMemory> mMemory;
};