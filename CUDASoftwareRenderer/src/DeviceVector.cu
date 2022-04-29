#include<pch.h>
#include"DeviceVector.cuh"
#include<device_atomic_functions.h>
#include<device_launch_parameters.h>
#include<cuda.h>

void DeviceVector::Add(void* data, size_t size)
{
	size_t ptr = reinterpret_cast<size_t>(mVirtual) + mOffset;
	void* casted = reinterpret_cast<void*>(ptr);

	memcpy(casted, &data, size);
	//mOffset += size;
	atomicAdd(&mOffset, size);
	atomicAdd(&mCount, 1);
	//	mCount++;

	return;
}
