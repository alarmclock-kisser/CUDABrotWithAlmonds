
using ManagedCuda;
using ManagedCuda.BasicTypes;
using System.Runtime.InteropServices;

namespace CUDABrotWithAlmonds
{
	public class CudaMemoryHandling
	{
		// ----- ----- ATTRIBUTES ----- ----- \\
		private string Repopath;
		private ListBox LogList;
		private PrimaryContext Context;

		public List<CudaBuffer> Buffers = [];


		// ----- ----- CONSTRUCTORS ----- ----- \\
		public CudaMemoryHandling(string repopath, ListBox logList, PrimaryContext context)
		{
			this.Repopath = repopath;
			this.LogList = logList;
			this.Context = context;


		}





		// ----- ----- METHODS ----- ----- \\
		public string Log(string message = "", string inner = "", int indent = 0)
		{
			string indentString = new string(' ', indent);
			string logMessage = $"[Mem] {indentString}{message} ({inner})";
			this.LogList.Items.Add(logMessage);
			this.LogList.TopIndex = this.LogList.Items.Count - 1;
			return logMessage;
		}




		public void Dispose()
		{
			// Free buffers
		}

		public CudaBuffer? GetBuffer(IntPtr pointer)
		{
			// Find buffer obj by pointer
			CudaBuffer? obj = this.Buffers.FirstOrDefault(x => x.Pointer == pointer);
			if (obj == null)
			{
				// Log
				this.Log($"Buffer not found", pointer.ToString(), 1);
				
				return null;
			}

			return obj;
		}

		public long FreeBuffer(IntPtr pointer, bool readable = false)
		{
			// Get buffer
			CudaBuffer? obj = this.GetBuffer(pointer);
			if (obj == null)
			{
				return 0;
			}

			// Get size
			long size = this.GetBufferSize(pointer, readable);

			// Get device ptr
			CUdeviceptr ptr = new CUdeviceptr(pointer);

			// Free buffer
			this.Context.FreeMemory(ptr);

			return size;
		}

		public Type GetBufferType(IntPtr pointer)
		{
			Type defaultType = typeof(void);

			// Get buffer
			CudaBuffer? obj = this.GetBuffer(pointer);
			if (obj == null)
			{
				return defaultType;
			}

			return obj.Type;
		}

		public long GetBufferSize(IntPtr pointer, bool readable = false)
		{
			// Get buffer
			CudaBuffer? obj = this.GetBuffer(pointer);
			if (obj == null)
			{
				return 0;
			}

			// Get buffer type
			Type bufferType = obj.Type;

			// Get length in bytes
			long length = (long) obj.Length * Marshal.SizeOf(bufferType);

			// Make readable
			if (readable)
			{
				length /= 1024 * 1024;
			}

			return length;
		}

		public IntPtr PushData<T>(IEnumerable<T> data) where T : unmanaged
		{
			// Check data
			if (data == null || !data.Any())
			{
				this.Log("No data to push");
				return 0;
			}

			// Get length pointer
			IntPtr length = (nint) data.LongCount();

			// Allocate buffer & copy data
			CudaDeviceVariable<T> buffer = new CudaDeviceVariable<T>(length);
			buffer.CopyToDevice(data.ToArray());

			// Get pointer
			IntPtr pointer = buffer.DevicePointer.Pointer;

			// Log
			this.Log($"Pushed {length} bytes to device", pointer.ToString(), 1);

			// Create obj
			CudaBuffer obj = new()
			{
				Pointer = pointer,
				Length = length,
				Type = typeof(T)
			};

			// Add to dict
			this.Buffers.Add(obj);
			
			// Return pointer
			return pointer;
		}

		public T[] PullData<T>(IntPtr pointer, bool free = false) where T : unmanaged
		{
			// Get buffer
			CudaBuffer? obj = this.GetBuffer(pointer);
			if (obj == null || obj.Length == 0)
			{
				return [];
			}

			// Create array with long count
			T[] data = new T[(long) obj.Length];

			// Get device pointer
			CUdeviceptr ptr = new CUdeviceptr(pointer);

			// Copy data to host from device pointer
			this.Context.CopyToHost(data, ptr);

			// Log
			this.Log($"Pulled {obj.Length} bytes from device", pointer.ToString(), 1);

			// Free buffer
			if (free)
			{
				this.FreeBuffer(pointer);
			}

			// Return data
			return data;
		}

		public IntPtr AllocateBuffer<T>(IntPtr length) where T : unmanaged
		{
			// Allocate buffer
			CudaDeviceVariable<T> buffer = new(length);
			
			// Get pointer
			IntPtr pointer = buffer.DevicePointer.Pointer;
			
			// Log
			this.Log($"Allocated {length} bytes on device", pointer.ToString(), 1);
			
			// Create obj
			CudaBuffer obj = new()
			{
				Pointer = pointer,
				Length = length,
				Type = typeof(T)
			};

			// Add to dict
			this.Buffers.Add(obj);
			return pointer;
		}
	}



	public class CudaBuffer
	{
		// ----- ----- ATTRIBUTES ----- ----- \\
		public IntPtr Pointer { get; set; }
		public IntPtr Length { get; set; }
		public Type Type { get; set; } = typeof(void);







	}
}