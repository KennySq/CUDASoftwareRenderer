#include<pch.h>
#include"DeviceEntity.cuh"

DeviceEntity::DeviceEntity(void* virtualPtr, size_t size)
	: mVirtual(virtualPtr), mSize(size) {}

DeviceEntity::~DeviceEntity()
{
}
