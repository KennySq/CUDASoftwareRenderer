#include<pch.h>
#include"DeviceTexture.cuh"

DeviceTexture::DeviceTexture(void* virtualPtr, size_t size, unsigned int width, unsigned int height)
	: DeviceEntity(virtualPtr, size), mWidth(width), mHeight(height)
{
}

DeviceTexture::~DeviceTexture()
{
}