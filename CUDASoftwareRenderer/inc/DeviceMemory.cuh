#pragma once
#include"DeviceEntity.cuh"

struct DeviceMemory
{
public:
	DeviceMemory(long long initSize);
	~DeviceMemory();
	
	void* Alloc(size_t size)
	{
		size_t ptr = reinterpret_cast<size_t>(mVirtual) + mOffset;

		void* casted = reinterpret_cast<void*>(ptr);

		MemoryBlock block = MemoryBlock(casted, mOffset);

		mOffset += size;

		mMemoryBlocks.push_back(block);

		return casted;
	}


private:
	struct MemoryBlock
	{
		MemoryBlock(void* entity, size_t offset)
			: Virtual(entity), Offset(offset) {}

		size_t Offset;
		void* Virtual;
	};

	void* mVirtual;
	size_t mOffset;
	size_t mSize;

	std::vector<MemoryBlock> mMemoryBlocks;
};