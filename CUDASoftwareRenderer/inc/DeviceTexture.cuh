#pragma once
#include"DeviceEntity.cuh"
struct DeviceTexture : public DeviceEntity
{
	friend struct ResourceManager;
public:
	DeviceTexture(void* virtualPtr, size_t size, unsigned int width, unsigned int height);

	unsigned int GetWidth() const { return mWidth; }
	unsigned int GetHeight() const { return mHeight; }

	virtual ~DeviceTexture();
private:
	unsigned int mWidth;
	unsigned int mHeight;
};