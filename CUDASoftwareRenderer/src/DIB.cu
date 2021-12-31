#include<pch.h>
#include"DIB.cuh"
#include"DeviceTexture.cuh"

DIB::DIB(HWND hWnd, unsigned int width, unsigned int height)
	: mHandle(hWnd), mWidth(width), mHeight(height), mMemoryDC(nullptr), mRaw(nullptr)
{
	assert(hWnd != nullptr);

	HDC dc = GetDC(hWnd);
	mHandleDC = dc;
	assert(dc != nullptr);

	BITMAPINFO bitmapInfo{};

	bitmapInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	bitmapInfo.bmiHeader.biBitCount = 32;
	bitmapInfo.bmiHeader.biCompression = BI_RGB;
	bitmapInfo.bmiHeader.biHeight = height;
	bitmapInfo.bmiHeader.biWidth = width;
	bitmapInfo.bmiHeader.biPlanes = 1;

	mBitmap = CreateDIBSection(mMemoryDC, &bitmapInfo, DIB_RGB_COLORS, reinterpret_cast<void**>(&mRaw), nullptr, 0);
	mMemoryDC = CreateCompatibleDC(dc);

	assert(mMemoryDC != nullptr);
	
	SelectObject(mMemoryDC, mBitmap);

	assert(mBitmap != nullptr);

}

DIB::~DIB()
{
	ReleaseDC(mHandle, mMemoryDC);
}

void DIB::CopyBuffer(std::shared_ptr<DeviceTexture> virtualResource)
{
	cudaMemcpy(mRaw, virtualResource->GetVirtual(), mWidth * mHeight * sizeof(DWORD), cudaMemcpyDeviceToHost);
}

void DIB::Present()
{
	BitBlt(mHandleDC, 0, 0, mWidth, mHeight, mMemoryDC, 0, 0, SRCCOPY);
}
