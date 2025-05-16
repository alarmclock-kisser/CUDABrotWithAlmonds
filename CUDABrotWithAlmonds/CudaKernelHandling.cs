
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.NVRTC;
using ManagedCuda.VectorTypes;
using System.Diagnostics;

namespace CUDABrotWithAlmonds
{
	public class CudaKernelHandling
	{
		// ----- ----- ATTRIBUTES ----- ----- \\
		private string Repopath;
		private ListBox LogList;
		private PrimaryContext Context;
		private CudaMemoryHandling MemoryH;
		private ComboBox KernelsCombo;

		public CudaKernel? Kernel = null;
		public string? KernelName = null;
		public string? KernelFile = null;
		public string? KernelCode = null;


		public List<string> SourceFiles => this.GetCuFiles();
		public List<string> CompiledFiles => this.GetPtxFiles();


		private string KernelPath => Path.Combine(this.Repopath, "Resources", "Kernels");

		// ----- ----- CONSTRUCTORS ----- ----- \\
		public CudaKernelHandling(string repopath, ListBox logList, PrimaryContext context, CudaMemoryHandling memoryH, ComboBox kernelsCombo)
		{
			this.Repopath = repopath;
			this.LogList = logList;
			this.Context = context;
			this.MemoryH = memoryH;
			this.KernelsCombo = kernelsCombo;

			// Register events
			// this.KernelsCombo.SelectedIndexChanged += (s, e) => this.LoadKernel(this.KernelsCombo.SelectedItem?.ToString() ?? "");

			// Compile all kernels
			this.CompileAll();

			// Fill kernels combobox
			this.FillKernelsCombo();
		}




		// ----- ----- METHODS ----- ----- \\
		public string Log(string message = "", string inner = "", int indent = 0)
		{
			string indentString = new string(' ', indent);
			string logMessage = $"[Krn] {indentString}{message} ({inner})";
			this.LogList.Items.Add(logMessage);
			this.LogList.TopIndex = this.LogList.Items.Count - 1;
			return logMessage;
		}


		public void Dispose()
		{
			// Dispose of kernels

		}

		public List<string> GetPtxFiles()
		{
			string path = Path.Combine(this.KernelPath, "PTX");

			// Get all PTX files in kernel path
			string[] files = Directory.GetFiles(path, "*.ptx").Select(f => Path.GetFullPath(f)).ToArray();

			// Return files
			return files.ToList();
		}

		public List<string> GetCuFiles()
		{
			string path = Path.Combine(this.KernelPath, "CU");

			// Get all CU files in kernel path
			string[] files = Directory.GetFiles(path, "*.cu").Select(f => Path.GetFullPath(f)).ToArray();

			// Return files
			return files.ToList();
		}

		public void CompileAll()
		{
			List<string> sourceFiles = this.SourceFiles;

			// Compile all source files
			foreach (string sourceFile in sourceFiles)
			{
				this.CompileKernel(sourceFile);
			}
		}

		public void FillKernelsCombo(int index = -1)
		{
			this.KernelsCombo.Items.Clear();

			// Get all PTX files in kernel path
			string[] files = this.CompiledFiles.Select(f => Path.GetFileNameWithoutExtension(f)).ToArray();

			// Add to combo box
			foreach (string file in files)
			{
				this.KernelsCombo.Items.Add(file);
			}

			// Select first item
			if (this.KernelsCombo.Items.Count > index)
			{
				this.KernelsCombo.SelectedIndex = index;
			}
		}

		public void SelectLatestKernel()
		{
			string[] files = this.CompiledFiles.ToArray();

			// Get file info (last modified), sort by last modified date, select latest
			FileInfo[] fileInfos = files.Select(f => new FileInfo(f)).OrderByDescending(f => f.LastWriteTime).ToArray();
			
			string latestFile = fileInfos.FirstOrDefault()?.FullName ?? "";
			string latestName = Path.GetFileNameWithoutExtension(latestFile) ?? "";
			this.KernelsCombo.SelectedItem = latestName;
		}

		public CudaKernel? LoadKernel(string kernelName, bool silent = false)
		{
			if (this.Context == null)
			{
				this.Log("No CUDA context available", "", 1);
				return null;
			}

			// Unload?
			if (this.Kernel != null)
			{
				this.UnloadKernel();
			}

			// Get kernel path
			string kernelPath = Path.Combine(this.KernelPath, "PTX", kernelName + ".ptx");

			// Get log path
			string logpath = Path.Combine(this.KernelPath, "Logs", kernelName + ".log");

			// Log
			Stopwatch sw = Stopwatch.StartNew();
			if (!silent)
			{
				this.Log("Started loading kernel " + kernelName);
			}

			// Try to load kernel
			try
			{
				// Load ptx code
				byte[] ptxCode = File.ReadAllBytes(kernelPath);

				string cuPath = Path.Combine(this.KernelPath, "CU", kernelName + ".cu");

				// Load kernel
				this.Kernel = this.Context.LoadKernelPTX(ptxCode, kernelName);
				this.KernelName = kernelName;
				this.KernelFile = kernelPath;
				this.KernelCode = File.ReadAllText(cuPath);
			}
			catch (Exception ex)
			{
				if (!silent)
				{
					this.Log("Failed to load kernel " + kernelName, ex.Message, 1);
					string logMsg = ex.Message + Environment.NewLine + Environment.NewLine + ex.InnerException?.Message ?? "";
					File.WriteAllText(logpath, logMsg);
				}
				this.Kernel = null;
			}

			// Log
			sw.Stop();
			long deltaMicros = sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L));
			if (!silent)
			{
				this.Log("Kernel loaded within " + deltaMicros.ToString("N0") + " µs", "", 2);
			}

			return this.Kernel;
		}

		public void UnloadKernel()
		{
			// Unload kernel
			if (this.Kernel != null)
			{
				this.Context.UnloadKernel(this.Kernel);
				this.Kernel = null;
			}
		}

		public string CompileKernel(string filepath)
		{
			if (this.Context == null)
			{
				this.Log("No CUDA context available", "", 1);
				return "";
			}

			// If file is not a .cu file, but raw kernel string, compile that
			if (Path.GetExtension(filepath) != ".cu")
			{
				return this.CompileString(filepath);
			}

			string kernelName = Path.GetFileNameWithoutExtension(filepath);

			string logpath = Path.Combine(this.KernelPath, "Logs", kernelName + ".log");

			Stopwatch sw = Stopwatch.StartNew();
			this.Log("Compiling kernel " + kernelName);

			// Load kernel file
			string kernelCode = File.ReadAllText(filepath);


			CudaRuntimeCompiler rtc = new(kernelCode, kernelName);

			try
			{
				// Compile kernel
				rtc.Compile([]);

				if (rtc.GetLogAsString().Length > 0)
				{
					this.Log("Kernel compiled with warnings", "", 1);
					File.WriteAllText(logpath, rtc.GetLogAsString());
				}


				sw.Stop();
				long deltaMicros = sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L));
				this.Log("Kernel compiled within " + deltaMicros.ToString("N0") + " µs", "Repo\\" + Path.GetRelativePath(this.Repopath, logpath), 2);

				// Get ptx code
				byte[] ptxCode = rtc.GetPTX();

				// Export ptx
				string ptxPath = Path.Combine(this.KernelPath, "PTX", kernelName + ".ptx");
				File.WriteAllBytes(ptxPath, ptxCode);

				this.Log("PTX code exported to " + ptxPath, "", 1);

				return ptxPath;
			}
			catch (Exception ex)
			{
				File.WriteAllText(logpath, rtc.GetLogAsString());
				this.Log(ex.Message, ex.InnerException?.Message ?? "", 1);

				return "";
			}

		}

		public string CompileString(string kernelString)
		{
			if (this.Context == null)
			{
				this.Log("No CUDA context available", "", 1);
				return "";
			}

			string kernelName = kernelString.Split("void ")[1].Split("(")[0];

			string logpath = Path.Combine(this.KernelPath, "Logs", kernelName + ".log");

			Stopwatch sw = Stopwatch.StartNew();
			this.Log("Compiling kernel " + kernelName);

			// Load kernel file
			string kernelCode = kernelString;

			// Save also the kernel string as .c file
			string cPath = Path.Combine(this.KernelPath, "CU", kernelName + ".cu");
			File.WriteAllText(cPath, kernelCode);


			CudaRuntimeCompiler rtc = new(kernelCode, kernelName);

			try
			{
				// Compile kernel
				rtc.Compile([]);

				if (rtc.GetLogAsString().Length > 0)
				{
					this.Log("Kernel compiled with warnings", "", 1);
					File.WriteAllText(logpath, rtc.GetLogAsString());
				}


				sw.Stop();
				long deltaMicros = sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L));
				this.Log("Kernel compiled in " + deltaMicros.ToString("N0") + " µs", "Repo\\" + Path.GetRelativePath(this.Repopath, logpath), 2);

				// Get ptx code
				byte[] ptxCode = rtc.GetPTX();

				// Export ptx
				string ptxPath = Path.Combine(this.KernelPath, "PTX", kernelName + ".ptx");
				File.WriteAllBytes(ptxPath, ptxCode);

				this.Log("PTX code exported to " + ptxPath, "", 1);

				return ptxPath;
			}
			catch (Exception ex)
			{
				File.WriteAllText(logpath, rtc.GetLogAsString());
				this.Log(ex.Message, ex.InnerException?.Message ?? "", 1);

				return "";
			}
		}

		public IntPtr ExecuteKernel(IntPtr pointer, int width, int height, int channels, int bitdepth, object[] arguments)
		{
			// Check if kernel is loaded
			if (this.Kernel == null)
			{
				this.Log($"Kernel not loaded", this.KernelName ?? "N/A", 1);
				return pointer;
			}

			// Get arguments
			Dictionary<string, Type> args = this.GetArguments();

			// Get pointer
			CUdeviceptr devicePtr = new(pointer);

			// Allocate output buffer
			CUdeviceptr outputPtr = new(this.MemoryH.AllocateBuffer<byte>(width * height * ((channels * bitdepth) / 8)));

			// Merge arguments with invariables
			object[] kernelArgs = this.MergeArguments(devicePtr, outputPtr, width, height, channels, bitdepth, arguments);

			// Für ein 4-Kanal-Bild (RGBA): pixelIndex = (y * width + x) * 4;
			int totalThreadsX = width;
			int totalThreadsY = height;

			// Blockgröße (z. B. 16×16 Threads pro Block)
			int blockSizeX = 8;
			int blockSizeY = 8;

			// Gridgröße = Gesamtgröße / Blockgröße (aufrunden)
			int gridSizeX = (totalThreadsX + blockSizeX - 1) / blockSizeX;
			int gridSizeY = (totalThreadsY + blockSizeY - 1) / blockSizeY;

			this.Kernel.BlockDimensions = new dim3(blockSizeX, blockSizeY, 1);  // 2D-Block
			this.Kernel.GridDimensions = new dim3(gridSizeX, gridSizeY, 1);     // 2D-Grid


			// Run with arguments
			this.Kernel.Run(kernelArgs);

			this.Log($"Kernel executed", this.KernelName ?? "N/A", 1);

			// Synchronize
			this.Context.Synchronize();

			// Return output pointer
			return outputPtr.Pointer;
		}

		public Type GetArgumentType(string typeName)
		{
			// Pointers are always IntPtr (containing *)
			if (typeName.Contains("*"))
			{
				return typeof(IntPtr);
			}

			string typeIdentifier = typeName.Split(' ').LastOrDefault()?.Trim() ?? "void";
			Type type = typeIdentifier switch
			{
				"int" => typeof(int),
				"float" => typeof(float),
				"double" => typeof(double),
				"char" => typeof(char),
				"bool" => typeof(bool),
				"void" => typeof(void),
				"byte" => typeof(byte),
				_ => typeof(void)
			};

			return type;
		}

		public Dictionary<string, Type> GetArguments(string? kernelCode = null)
		{
			kernelCode ??= this.KernelCode;
			if (string.IsNullOrEmpty(kernelCode) || this.Kernel == null)
			{
				this.Log($"Kernel not loaded", this.KernelName ?? "N/A", 1);
				return [];
			}

			Dictionary<string, Type> arguments = [];

			int index = kernelCode.IndexOf("__global__ void");
			if (index == -1)
			{
				this.Log($"'__global__ void' not found", this.KernelName ?? "N/A", 1);
				return [];
			}

			index = kernelCode.IndexOf("(", index);
			if (index == -1)
			{
				this.Log($"'(' not found", this.KernelName ?? "N/A", 1);
				return [];
			}

			int endIndex = kernelCode.IndexOf(")", index);
			if (endIndex == -1)
			{
				this.Log($"')' not found", this.KernelName ?? "N/A", 1);
				return [];
			}

			string[] args = kernelCode.Substring(index + 1, endIndex - index - 1).Split(',').Select(x => x.Trim()).ToArray();

			// Get loaded kernels function args
			for (int i = 0; i < args.Length; i++)
			{
				string name = args[i].Split(' ').LastOrDefault() ?? "N/A";
				string typeName = args[i].Replace(name, "").Trim();
				Type type = this.GetArgumentType(typeName);

				// Add to dictionary
				arguments.Add(name, type);
			}

			return arguments;
		}

		public object[] MergeArguments(CUdeviceptr inputPointer, CUdeviceptr outputPointer, int width, int height, int channels, int bitdepth, object[] arguments)
		{
			// Get kernel argument definitions
			Dictionary<string, Type> args = this.GetArguments();

			// Create array for kernel arguments
			object[] kernelArgs = new object[args.Count];

			// Integrate invariables if name fits (contains)
			for (int i = 0; i < kernelArgs.Length; i++)
			{
				string name = args.ElementAt(i).Key;
				Type type = args.ElementAt(i).Value;

				if (name.Contains("input") && type == typeof(IntPtr))
				{
					kernelArgs[i] = inputPointer;
					this.Log($"Input pointer: {inputPointer}", "", 1);
				}
				else if (name.Contains("output") && type == typeof(IntPtr))
				{
					kernelArgs[i] = outputPointer;
					this.Log($"Output pointer: {outputPointer}", "", 1);
				}
				else if (name.Contains("width") && type == typeof(int))
				{
					kernelArgs[i] = width;
					this.Log($"Width: {width}", "", 1);
				}
				else if (name.Contains("height") && type == typeof(int))
				{
					kernelArgs[i] = height;
					this.Log($"Height: {height}", "", 1);
				}
				else if (name.Contains("channel") && type == typeof(int))
				{
					kernelArgs[i] = channels;
					this.Log($"Channels: {channels}", "", 1);
				}
				else if (name.Contains("bit") && type == typeof(int))
				{
					kernelArgs[i] = bitdepth;
					this.Log($"Bitdepth: {bitdepth}", "", 1);
				}
				else
				{
					// Check if argument is in arguments array
					for (int j = 0; j < arguments.Length; j++)
					{
						if (name == args.ElementAt(j).Key)
						{
							kernelArgs[i] = arguments[j];
							break;
						}
					}

					// If not found, set to 0
					if (kernelArgs[i] == null)
					{
						kernelArgs[i] = 0;
					}
				}
			}

			// DEBUG LOG
			this.Log("Kernel arguments: " + string.Join(", ", kernelArgs.Select(x => x.ToString())), "", 1);

			// Return kernel arguments
			return kernelArgs;
		}
	}
}