#pragma once
#include"DeviceEntity.cuh"

struct DeviceMemory
{
public:
	DeviceMemory(size_t initSize);
	
	
	std::shared_ptr<DeviceEntity> Alloc(size_t size);

private:
	struct MemoryBlock
	{
		MemoryBlock(std::shared_ptr<DeviceEntity> entity, size_t offset)
			: Entity(entity), Offset(offset) {}

		size_t Offset;
		std::shared_ptr<DeviceEntity> Entity;
	};

	void* mVirtual;
	size_t mOffset;
	size_t mSize;

	std::vector<MemoryBlock> mMemoryBlocks;
};