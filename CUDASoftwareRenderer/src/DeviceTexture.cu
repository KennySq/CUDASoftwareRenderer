#include<pch.h>
#include"DeviceTexture.cuh"

DeviceTexture::DeviceTexture(void* virtualPtr, size_t size)
	: DeviceEntity(virtualPtr, size), mWidth(0), mHeight(0)
{
}

DeviceTexture::~DeviceTexture()
{
}