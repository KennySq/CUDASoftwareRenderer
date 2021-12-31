#include<pch.h>
#include"ResourceManager.cuh"
#include"DeviceMemory.cuh"
#include"DeviceTexture.cuh"

ResourceManager::ResourceManager()
	: mMemory(std::make_unique<DeviceMemory>(-1))
{
}

std::shared_ptr<DeviceTexture> ResourceManager::CreateTexture2D(unsigned int width, unsigned height)
{
	size_t byteSize = width * height * sizeof(DWORD);

	std::shared_ptr<DeviceTexture> texture = std::static_pointer_cast<DeviceTexture, DeviceEntity>(mMemory->Alloc(byteSize));
	
	texture->mWidth = width;
	texture->mHeight = height;

	return texture;
}