#include<pch.h>
#include"DeviceTexture.cuh"

DeviceTexture::DeviceTexture(void* virtualPtr, size_t size)
	: DeviceEntity(virtualPtr, size)
{
}

DeviceTexture::~DeviceTexture()
{
}