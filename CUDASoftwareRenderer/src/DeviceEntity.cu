#include<pch.h>
#include"DeviceEntity.cuh"

DeviceEntity::DeviceEntity(void* virtualPtr, unsigned int size)
	: mVirtual(virtualPtr), mSize(size) {}
