#pragma once
#include"DeviceEntity.cuh"

struct DeviceBuffer : public DeviceEntity
{
	friend struct ResourceManager;
public:
	DeviceBuffer(void* virtualPtr, size_t size);
	virtual ~DeviceBuffer();

	unsigned int GetStride() const { return mStride; }

private:
	unsigned int mStride;
};
