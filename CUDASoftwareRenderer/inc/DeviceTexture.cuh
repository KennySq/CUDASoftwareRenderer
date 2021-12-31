#pragma once
#include"DeviceEntity.cuh"
struct DeviceTexture : public DeviceEntity
{
	friend struct ResourceManager;
public:
private:
	unsigned int mWidth;
	unsigned int mHeight;
};