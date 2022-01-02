#pragma once
#include"DeviceEntity.cuh"
struct DeviceTexture : public DeviceEntity
{
	friend struct ResourceManager;
public:
	DeviceTexture(void* virtualPtr, size_t size);
	virtual ~DeviceTexture();
private:
	unsigned int mWidth;
	unsigned int mHeight;
};