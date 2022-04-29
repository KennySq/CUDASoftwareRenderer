#pragma once

class DeviceVector
{
public:
	__device__ DeviceVector(size_t count, size_t stride)
		: mSize(stride * count), mCount(0), mVirtual(malloc(stride * count)), mOffset(0)
	{
	}
	__device__ ~DeviceVector()
	{

	}
	__device__ void Add(void* data, size_t size)
	{
		size_t ptr = reinterpret_cast<size_t>(mVirtual) + mOffset;
		void* casted = reinterpret_cast<void*>(ptr);

		memcpy(casted, &data, size);
		mOffset += size;
		mCount++;

		return;
	}

	__device__ void* GetData()
	{
		return mVirtual;
	}

	__device__ void Clear()
	{
		memset(mVirtual, 0xCD, mOffset);
		mOffset = 0;
		mCount = 0;
	}

	__device__ size_t GetCount() const
	{
		return mCount;
	}

private:
	void* mVirtual;
	size_t mOffset;
	size_t mSize;
	size_t mCount;
};