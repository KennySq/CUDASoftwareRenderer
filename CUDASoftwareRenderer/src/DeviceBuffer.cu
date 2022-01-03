#include<pch.h>
#include"DeviceBuffer.cuh"

DeviceBuffer::DeviceBuffer(void* virtualPtr, size_t size)
	: DeviceEntity(virtualPtr, size)
{
}

DeviceBuffer::~DeviceBuffer()
{
}