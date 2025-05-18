using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace CUDABrotWithAlmonds
{
	public partial class WindowMain : Form
	{
		// ----- ----- ATTRIBUTES ----- ----- \\
		public string Repopath;

		private ImageHandling ImageH;
		private ImageRecorder Recorder;

		private CudaContextHandling ContextH;

		private GuiBuilder GuiB;

		public List<string> ExplorationKernelSubstrings = new()
		{
			"mandelbrot",
			"newton",
			"julia",
			"burning",
			"tricorn",
			"multibrot"
		};

		private bool MandelbrotMode = false;
		private bool isDragging = false;
		private Point mouseDownLocation = new(0, 0);
		private bool ctrlKeyPressed;
		private bool kernelExecutionRequired;
		private float mandelbrotZoomFactor = 1.0f;
		private Stopwatch stopwatch = new();
		private bool isProcessing;
		private Dictionary<NumericUpDown, int> previousNumericValues = [];

		private Form? fullScreenForm = null;
		private Dictionary<string, object>? currentOverlayArgs = null;

		// ----- ----- CONSTRUCTORS ----- ----- \\
		public WindowMain()
		{
			this.InitializeComponent();

			// Set repopath
			this.Repopath = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"..\..\..\"));

			// Window position
			this.StartPosition = FormStartPosition.Manual;
			this.Location = new Point(0, 0);

			// Init. classes
			this.Recorder = new ImageRecorder(this.Repopath, this.label_cached);
			this.ImageH = new ImageHandling(this.Repopath, this.listBox_images, this.pictureBox_view, null, this.label_meta);
			this.ContextH = new CudaContextHandling(this.Repopath, this.listBox_log, this.comboBox_devices, this.comboBox_kernels, this.progressBar_vram);
			this.GuiB = new GuiBuilder(this.Repopath, this.listBox_log, this.ContextH, this.ImageH, this.panel_kernel);

			// Register events
			this.listBox_images.DoubleClick += (s, e) => this.MoveImage(this.listBox_images.SelectedIndex);
			this.listBox_log.MouseHover += (s, e) => this.ShowLogTooltip();
			this.comboBox_kernels.SelectedIndexChanged += (s, e) => this.LoadKernel(this.comboBox_kernels.SelectedItem?.ToString() ?? "");
			this.RegisterNumericToSecondPow(this.numericUpDown_size);
			this.RegisterNumericToSecondPow(this.numericUpDown_createSize);


			// Fill devices combobox
			this.ContextH.FillDevicesCombobox();

			// Select first device
			if (this.comboBox_devices.Items.Count > 0)
			{
				this.comboBox_devices.SelectedIndex = 0;
			}

			// Set latest kernel
			// this.ContextH.KernelH?.SelectLatestKernel();
		}





		// ----- ----- METHODS ----- ----- \\
		public void ReselectImage()
		{
			int index = this.listBox_images.SelectedIndex;

			this.ImageH.FillImagesListBox();

			if (index >= 0 && index < this.listBox_images.Items.Count)
			{
				this.listBox_images.SelectedIndex = index;
			}
			else
			{
				this.listBox_images.SelectedIndex = -1;
			}

		}

		public void ShowLogTooltip(int index = -1)
		{
			if (index == -1 && this.listBox_log.SelectedIndex != -1)
			{
				index = this.listBox_log.SelectedIndex;
			}
			if (index < 0 || index >= this.listBox_log.Items.Count)
			{
				return;
			}
			string message = this.listBox_log.Items[index].ToString() ?? "N/A";
			ToolTip tooltip = new();
			tooltip.Show(message, this.listBox_log, 0, 0, 10000);
		}

		public void MoveImage(int index = -1, bool refresh = true)
		{
			if (index == -1 && this.ImageH.CurrentObject != null)
			{
				index = this.ImageH.Images.IndexOf(this.ImageH.CurrentObject);
			}

			if (index < 0 && index >= this.ImageH.Images.Count)
			{
				MessageBox.Show("Invalid index", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			ImageObject image = this.ImageH.Images[index];

			// Move image Host <-> CUDA
			if (image.OnHost)
			{
				// Move to CUDA: Get bytes
				byte[] bytes = image.GetPixelsAsBytes();

				// STOPWATCH
				Stopwatch sw = Stopwatch.StartNew();

				// Create buffer
				IntPtr pointer = this.ContextH.MemoryH?.PushData(bytes, this.checkBox_silent.Checked) ?? 0;

				// STOPWATCH
				sw.Stop();
				this.label_pushTime.Text = $"Push time: {sw.ElapsedMilliseconds} ms";

				// Check pointer
				if (pointer == 0)
				{
					MessageBox.Show("Failed to push data to CUDA", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
					return;
				}

				// Set pointer, void image
				image.Pointer = pointer;
				image.Img = null;
			}
			else if (image.OnDevice)
			{
				// STOPWATCH
				Stopwatch sw = Stopwatch.StartNew();

				// Move to Host
				byte[] bytes = this.ContextH.MemoryH?.PullData<byte>(image.Pointer, true, this.checkBox_silent.Checked) ?? [];
				if (bytes.Length == 0)
				{
					MessageBox.Show("Failed to pull data from CUDA", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
					return;
				}

				// STOPWATCH
				sw.Stop();
				this.label_pullTime.Text = $"Pull time: {sw.ElapsedMilliseconds} ms";

				// Create image
				image.SetImageFromBytes(bytes, true);
			}

			// Refill list
			if (refresh)
			{
				this.ImageH.FillImagesListBox();
			}
		}

		public void LoadKernel(string kernelName = "")
		{
			// Load kernel
			this.ContextH.KernelH?.LoadKernel(kernelName, this.checkBox_silent.Checked);
			if (this.ContextH.KernelH?.Kernel == null)
			{
				MessageBox.Show("Failed to load kernel", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			// Toggle mandelbrot mode if kernel name contains any of the exploration substrings
			bool isExplorationMode = this.ExplorationKernelSubstrings.Any(sub => kernelName.Contains(sub, StringComparison.OrdinalIgnoreCase));

			this.ToggleMandelbrotMode(isExplorationMode);

			// Get arguments
			this.GuiB.BuildPanel(0.55f, this.checkBox_optionalArgsOnly.Checked);

		}

		public void ExecuteKernelOOP(int index = -1, string kernelName = "", bool addMod = false)
		{
			// If index is -1, use current object
			if (index == -1 && this.ImageH.CurrentObject != null)
			{
				index = this.ImageH.Images.IndexOf(this.ImageH.CurrentObject);
			}

			if (index < 0 || index >= this.ImageH.Images.Count)
			{
				MessageBox.Show("Invalid index", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			// Get image
			ImageObject? image = this.ImageH.Images[index];
			if (image == null)
			{
				MessageBox.Show("Invalid image", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			// Verify image on device
			bool moved = false;
			if (image.OnHost)
			{
				this.MoveImage(index);
				moved = true;
				if (image.OnHost)
				{
					MessageBox.Show("Could not move image to device", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
					return;
				}
			}

			// STOPWATCH
			Stopwatch sw = Stopwatch.StartNew();

			// Load kernel
			this.ContextH.KernelH?.LoadKernel(kernelName, this.checkBox_silent.Checked);
			if (this.ContextH.KernelH?.Kernel == null)
			{
				MessageBox.Show("Failed to load kernel", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			// If kernel is Exporable, set mode
			bool isExplorationMode = this.ExplorationKernelSubstrings.Any(sub => kernelName.Contains(sub, StringComparison.OrdinalIgnoreCase));
			this.ToggleMandelbrotMode(isExplorationMode);

			// Get image attributes for kernel call
			IntPtr pointer = image.Pointer;
			int width = image.Width;
			int height = image.Height;
			int channels = 4;
			int bitdepth = image.BitsPerPixel / channels;

			// Get variable arguments
			object[] args = this.GuiB.GetArgumentValues();

			// Call exec kernel -> pointer
			image.Pointer = this.ContextH.KernelH?.ExecuteKernel(pointer, width, height, channels, bitdepth, args, this.checkBox_silent.Checked) ?? image.Pointer;

			// STOPWATCH
			sw.Stop();
			this.label_execTime.Text = $"Exec. time: {sw.ElapsedMilliseconds} ms";

			// Optional: Move back to host
			if (moved)
			{
				this.MoveImage(index);
			}

			// If kernel is Mandelbrot, cache image with interval
			if (this.MandelbrotMode && image.Img != null && (this.checkBox_record.Checked || IsKeyLocked(Keys.CapsLock)))
			{
				this.Recorder.AddImage(image.Img, this.stopwatch.ElapsedMilliseconds);
				this.stopwatch.Restart(); // Restart stopwatch
			}

			// Reset cache if checkbox is unchecked || cache isnt empty || not CAPS locked
			if (!this.checkBox_record.Checked && this.Recorder.CachedImages.Count != 0 && !IsKeyLocked(Keys.CapsLock))
			{
				this.Recorder.CachedImages.Clear();
				this.Recorder.CountLabel.Text = $"Images: -";
			}

			// Add modification to image
			if (addMod)
			{
				image.AddModification(this.ContextH.KernelH?.KernelName ?? "Unknown", args.LastOrDefault()?.ToString() ?? "");
			}

			// Refill list
			this.ImageH.FillImagesListBox();
		}

		private void ToggleMandelbrotMode(bool? forceMode = null)
		{
			if (forceMode == null)
			{
				this.MandelbrotMode = !this.MandelbrotMode;
			}
			else
			{

				this.MandelbrotMode = forceMode.Value;
			}

			// 3. Alle bestehenden Event-Handler entfernen (sauberer Reset)
			this.pictureBox_view.MouseDown -= this.ImageH.ViewPBox_MouseDown;
			this.pictureBox_view.MouseMove -= this.ImageH.ViewPBox_MouseMove;
			this.pictureBox_view.MouseUp -= this.ImageH.ViewPBox_MouseUp;
			this.pictureBox_view.MouseWheel -= this.ImageH.ViewPBox_MouseWheel;

			this.pictureBox_view.MouseDown -= this.pictureBox_view_MouseDown;
			this.pictureBox_view.MouseMove -= this.pictureBox_view_MouseMove;
			this.pictureBox_view.MouseUp -= this.pictureBox_view_MouseUp;
			this.pictureBox_view.MouseWheel -= this.pictureBox_view_MouseWheel;

			// 4. Neue Event-Handler registrieren
			if (this.MandelbrotMode)
			{
				this.stopwatch = Stopwatch.StartNew(); // Start stopwatch
				this.pictureBox_view.MouseDown += this.pictureBox_view_MouseDown;
				this.pictureBox_view.MouseMove += this.pictureBox_view_MouseMove;
				this.pictureBox_view.MouseUp += this.pictureBox_view_MouseUp;
				this.pictureBox_view.MouseWheel += this.pictureBox_view_MouseWheel;
			}
			else
			{
				this.Recorder.ResetCache(); // Reset cache
				this.stopwatch.Stop(); // Stop stopwatch
				this.pictureBox_view.MouseDown += this.ImageH.ViewPBox_MouseDown;
				this.pictureBox_view.MouseMove += this.ImageH.ViewPBox_MouseMove;
				this.pictureBox_view.MouseUp += this.ImageH.ViewPBox_MouseUp;
				this.pictureBox_view.MouseWheel += this.ImageH.ViewPBox_MouseWheel;
			}
		}

		public void ExecuteKernelOOPAll(string kernelName = "")
		{
			int count = this.ImageH.Images.Count;
			if (count == 0)
			{
				MessageBox.Show("No images to process", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			// Load kernel
			this.ContextH.KernelH?.LoadKernel(kernelName, this.checkBox_silent.Checked);
			if (this.ContextH.KernelH?.Kernel == null)
			{
				MessageBox.Show("Failed to load kernel", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			// Move all images to device
			for (int i = 0; i < count; i++)
			{
				this.MoveImage(i, false);
			}

			// Execute kernel on all images
			for (int i = 0; i < count; i++)
			{
				this.ExecuteKernelOOP(i, kernelName);
			}

			// Move all images to host
			for (int i = 0; i < count; i++)
			{
				this.MoveImage(i, false);
			}

			// Refill list
			this.ImageH.FillImagesListBox();
		}

		public void RenderArgsIntoPictureboxOld()
		{
			object[] argValues = this.GuiB.GetArgumentValues();
			string[] argNames = this.GuiB.GetArgumentNames();
			Dictionary<string, object> args = argValues
				.Select((value, index) => new { Name = argNames[index], Value = value })
				.ToDictionary(x => x.Name, x => x.Value);

			Size size = new(200, 100);
			Point point = new(0, 0);

			// Remove pointers & dimensions from args
			args = args.Where(x => !x.Key.ToLower().Contains("input") && !x.Key.ToLower().Contains("output") && !x.Key.ToLower().Contains("pixel") && !x.Key.ToLower().Contains("width") && !x.Key.ToLower().Contains("height")).ToDictionary(x => x.Key, x => x.Value);

			// Render args into picture
			this.GuiB.RenderOverlayInPicturebox(this.ImageH.ViewPBox, args, -1, Color.White, null, null);
		}

		public void RenderArgsIntoPicturebox()
		{
			object[] argValues = this.GuiB.GetArgumentValues();
			string[] argNames = this.GuiB.GetArgumentNames();
			Dictionary<string, object> args = argValues
				.Select((value, index) => new { Name = argNames[index], Value = value })
				.ToDictionary(x => x.Name, x => x.Value);

			// Filter args (wie in deinem Code)
			args = args.Where(x => !x.Key.ToLower().Contains("input") &&
								  !x.Key.ToLower().Contains("output") &&
								  !x.Key.ToLower().Contains("pixel") &&
								  !x.Key.ToLower().Contains("width") &&
								  !x.Key.ToLower().Contains("height"))
					   .ToDictionary(x => x.Key, x => x.Value);

			// Speichere args für Overlay-Zeichnung im Paint-Event
			this.currentOverlayArgs = args;

			// PictureBox neu zeichnen (ruft PictureBox_view_Paint auf)
			this.ImageH.ViewPBox.Invalidate();
		}

		public void ExportAllGif()
		{
			List<Image> images = this.ImageH.Images.Where(x => x.Img != null).Select(x => x.Img ?? new Bitmap(1, 1)).ToList();
			if (images.Count == 0)
			{
				MessageBox.Show("No images to export", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			this.Recorder.ResetCache(); // Reset cache
			this.Recorder.CachedImages = images;
			this.Recorder.CountLabel.Text = $"Images: {this.Recorder.CachedImages.Count}";

			// Create GIF
			this.button_createGif_Click(this, EventArgs.Empty);
		}

		public void RegisterNumericToSecondPow(NumericUpDown numeric)
		{
			// Initialwert speichern
			this.previousNumericValues.Add(numeric, (int) numeric.Value);

			numeric.ValueChanged += (s, e) =>
			{
				// Verhindere rekursive Aufrufe
				if (this.isProcessing)
				{
					return;
				}

				this.isProcessing = true;

				try
				{
					int newValue = (int) numeric.Value;
					int oldValue = this.previousNumericValues[numeric];
					int max = (int) numeric.Maximum;
					int min = (int) numeric.Minimum;

					// Nur verarbeiten, wenn sich der Wert tats chlich ge ndert hat
					if (newValue != oldValue)
					{
						int calculatedValue;

						if (newValue > oldValue)
						{
							// Verdoppeln, aber nicht  ber Maximum
							calculatedValue = Math.Min(oldValue * 2, max);
						}
						else if (newValue < oldValue)
						{
							// Halbieren, aber nicht unter Minimum
							calculatedValue = Math.Max(oldValue / 2, min);
						}
						else
						{
							calculatedValue = oldValue;
						}

						// Nur aktualisieren wenn notwendig
						if (calculatedValue != newValue)
						{
							numeric.Value = calculatedValue;
						}

						this.previousNumericValues[numeric] = calculatedValue;
					}
				}
				finally
				{
					this.isProcessing = false;
				}
			};
		}



		// Mandelbrot events
		private void pictureBox_view_MouseDown(object? sender, MouseEventArgs e)
		{
			if (e.Button == MouseButtons.Left)
			{
				this.isDragging = true;
				this.mouseDownLocation = e.Location;
			}
		}

		private void pictureBox_view_MouseMove(object? sender, MouseEventArgs e)
		{
			if (!this.isDragging || this.ImageH.CurrentObject == null)
			{
				return;
			}

			try
			{
				// 1. Find NumericUpDown controls more efficiently
				NumericUpDown? numericX = this.panel_kernel.Controls.OfType<NumericUpDown>().FirstOrDefault(x => x.Name.ToLower().Contains("offsetx"));
				NumericUpDown? numericY = this.panel_kernel.Controls.OfType<NumericUpDown>().FirstOrDefault(x => x.Name.ToLower().Contains("offsety"));
				NumericUpDown? numericZ = this.panel_kernel.Controls.OfType<NumericUpDown>().FirstOrDefault(x => x.Name.ToLower().Contains("zoom"));
				NumericUpDown? numericMouseX = this.panel_kernel.Controls.OfType<NumericUpDown>().FirstOrDefault(x => x.Name.ToLower().Contains("mousex"));
				NumericUpDown? numericMouseY = this.panel_kernel.Controls.OfType<NumericUpDown>().FirstOrDefault(x => x.Name.ToLower().Contains("mousey"));

				if (!(numericX == null || numericY == null || numericZ == null))
				{
					// 2. Calculate smooth delta with sensitivity factor and zoom
					float sensitivity = 0.001f * (float) (1 / numericZ.Value);
					decimal deltaX = (decimal) ((e.Location.X - this.mouseDownLocation.X) * -sensitivity);
					decimal deltaY = (decimal) ((e.Location.Y - this.mouseDownLocation.Y) * -sensitivity);

					// 3. Update values with boundary checks
					this.UpdateNumericValue(numericX, deltaX);
					this.UpdateNumericValue(numericY, deltaY);
				}

				// 4. Update mouse position for smoother continuous dragging
				this.mouseDownLocation = e.Location;

				// 5. Update mouse coordinates in NumericUpDown controls
				if (!(numericMouseX == null || numericMouseY == null))
				{
					this.UpdateNumericValue(numericMouseX, e.Location.X);
					this.UpdateNumericValue(numericMouseY, e.Location.Y);
				}
			}
			catch (Exception ex)
			{
				Debug.WriteLine($"MouseMove error: {ex.Message}");
			}
		}

		private void UpdateNumericValue(NumericUpDown numeric, decimal delta)
		{
			decimal newValue = numeric.Value + delta;

			// Ensure value stays within allowed range
			if (newValue < numeric.Minimum)
			{
				newValue = numeric.Minimum;
			}

			if (newValue > numeric.Maximum)
			{
				newValue = numeric.Maximum;
			}

			numeric.Value = newValue;
		}

		private void pictureBox_view_MouseUp(object? sender, MouseEventArgs e)
		{
			if (e.Button == MouseButtons.Left)
			{
				this.isDragging = false;

				// Re-execute kernel
				this.button_executeOOP_Click(sender, e);

				this.RenderArgsIntoPicturebox();
			}
		}

		private void pictureBox_view_MouseWheel(object? sender, MouseEventArgs e)
		{
			// Check for CTRL key press
			if (Control.ModifierKeys == Keys.Control)
			{
				this.ctrlKeyPressed = true;
				this.kernelExecutionRequired = true; // Set the flag
				NumericUpDown? numericI = this.panel_kernel.Controls.OfType<NumericUpDown>().FirstOrDefault(x => x.Name.ToLower().Contains("iter"));
				if (numericI == null)
				{
					this.ContextH.KernelH?.Log("MaxIter control not found!", "", 3);
					return;
				}

				// Increase/Decrease maxIter
				if (e.Delta > 0)
				{
					numericI.Value += 2;
				}
				else if (e.Delta < 0)
				{
					if (numericI.Value > 0)
					{
						numericI.Value -= 2;
					}
				}
				return;
			}

			// Check if CTRL key was previously pressed
			if (this.ctrlKeyPressed)
			{
				this.ctrlKeyPressed = false; // Reset the flag
				this.kernelExecutionRequired = true;
			}

			// 1. Find NumericUpDown controls more efficiently
			NumericUpDown? numericZ = this.panel_kernel.Controls.OfType<NumericUpDown>().FirstOrDefault(x => x.Name.ToLower().Contains("zoom"));
			if (numericZ == null)
			{
				this.ContextH.KernelH?.Log("Zoom control not found!", "", 3);
				return;
			}

			// 2. Calculate zoom factor
			if (e.Delta > 0)
			{
				this.mandelbrotZoomFactor *= 1.1f;
			}
			else
			{
				this.mandelbrotZoomFactor /= 1.1f;
			}

			// 3. Update zoom value with boundary checks
			decimal newValue = (decimal) this.mandelbrotZoomFactor;
			if (newValue < numericZ.Minimum)
			{
				newValue = numericZ.Minimum;
			}
			if (newValue > numericZ.Maximum)
			{
				newValue = numericZ.Maximum;
			}
			numericZ.Value = newValue;

			// Call re-exec kernel
			this.kernelExecutionRequired = true;

			if (!this.ctrlKeyPressed && this.kernelExecutionRequired)
			{
				this.kernelExecutionRequired = false;
				this.button_executeOOP_Click(sender, e);
				this.RenderArgsIntoPicturebox();
			}
		}

		private void PictureBox_view_Paint(object? sender, PaintEventArgs e)
		{
			var pbox = sender as PictureBox;
			if (pbox == null) return;

			// Erstens: Basis-Image zeichnen (falls noch nicht automatisch)
			if (pbox.Image != null)
			{
				e.Graphics.DrawImage(pbox.Image, new Point(0, 0));
			}

			// Overlay nur zeichnen, wenn MandelbrotMode aktiv ist und Overlay-Daten vorhanden sind
			if (this.MandelbrotMode && currentOverlayArgs != null && currentOverlayArgs.Count > 0)
			{
				// Overlay erstellen - Größe auf PictureBox-Größe oder nach Wunsch
				Size overlaySize = new Size(pbox.Width / 8, pbox.Height / 8);

				// Overlay vom GuiBuilder holen (erwartet: CreateOverlayBitmap(Size, Dictionary, ...))
				Bitmap overlay = this.GuiB.CreateOverlayBitmap(overlaySize, currentOverlayArgs, fontSize: 12, color: Color.White, imageSize: pbox.Size);

				// Overlay transparent zeichnen an gewünschter Position (z.B. oben links)
				e.Graphics.DrawImageUnscaled(overlay, new Point(10, 10));

				overlay.Dispose();
			}
		}



		private Point TranslateMousePosToImageCoords(PictureBox pb, MouseEventArgs e)
		{
			if (pb.Image == null) return Point.Empty;

			var img = pb.Image;

			// Bild- und PictureBox-Verhältnisse
			float imageAspect = (float) img.Width / img.Height;
			float boxAspect = (float) pb.Width / pb.Height;

			int displayedWidth, displayedHeight;
			int offsetX, offsetY;

			if (boxAspect > imageAspect)
			{
				// Bild wird an Höhe angepasst
				displayedHeight = pb.Height;
				displayedWidth = (int) (imageAspect * displayedHeight);
				offsetX = (pb.Width - displayedWidth) / 2;
				offsetY = 0;
			}
			else
			{
				// Bild wird an Breite angepasst
				displayedWidth = pb.Width;
				displayedHeight = (int) (displayedWidth / imageAspect);
				offsetX = 0;
				offsetY = (pb.Height - displayedHeight) / 2;
			}

			// Mausposition relativ zum Bild
			int x = e.X - offsetX;
			int y = e.Y - offsetY;

			if (x < 0 || x >= displayedWidth || y < 0 || y >= displayedHeight)
			{
				// Maus außerhalb des Bildbereichs
				return Point.Empty;
			}

			// Skalierung zurückrechnen auf Originalbildgröße
			float scaleX = (float) img.Width / displayedWidth;
			float scaleY = (float) img.Height / displayedHeight;

			int imgX = (int) (x * scaleX);
			int imgY = (int) (y * scaleY);

			return new Point(imgX, imgY);
		}




		// ----- ----- EVENT HANDLERS ----- ----- \\
		private void button_import_Click(object sender, EventArgs e)
		{
			// If CTRL down, import GIF
			if (ModifierKeys == Keys.Control)
			{
				this.ImageH.ImportGif();
				return;
			}

			this.ImageH.ImportImage();
		}

		private void button_export_Click(object sender, EventArgs e)
		{
			// If CTRL down, export GIF
			if (ModifierKeys == Keys.Control)
			{
				this.ExportAllGif();
				return;
			}

			this.ImageH.CurrentObject?.Export(true);
		}

		private void button_createImage_Click(object sender, EventArgs e)
		{
			int dim = (int) this.numericUpDown_createSize.Value;
			Size size;

			// If CTRL down: Create empty image with screen size
			if (ModifierKeys == Keys.Control)
			{
				size = new Size(Screen.PrimaryScreen?.Bounds.Width ?? 1024, Screen.PrimaryScreen?.Bounds.Height ?? 1024);
			}
			else
			{
				size = new Size(dim, dim);
			}


			this.ImageH.CreateEmpty(Color.White, size, "");
			this.ImageH.FitZoom();
		}

		private void button_center_Click(object sender, EventArgs e)
		{
			// If CTRL down, fit zoom
			if (ModifierKeys == Keys.Control)
			{
				this.ImageH.CenterImage();
			}
			else
			{
				this.ImageH.FitZoom();
			}
		}

		private void button_executeOOP_Click(object? sender, EventArgs e)
		{
			string kernelName = this.comboBox_kernels.SelectedItem?.ToString() ?? "";

			// If CTRL down: Execute on all images
			if (ModifierKeys == Keys.Control)
			{
				this.ExecuteKernelOOPAll(kernelName);
				return;
			}

			bool addMod = this.ExplorationKernelSubstrings.Any(sub => kernelName.Contains(sub, StringComparison.OrdinalIgnoreCase));

			this.ToggleMandelbrotMode(addMod);

			this.ExecuteKernelOOP(-1, this.comboBox_kernels.SelectedItem?.ToString() ?? "", addMod);
		}

		private void button_reset_Click(object sender, EventArgs e)
		{
			this.ImageH.CurrentObject?.ResetImage();

			this.ImageH.FillImagesListBox();
		}

		private async void button_createGif_Click(object sender, EventArgs e)
		{
			// Folder MyPictures
			string folder = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyPictures), "CUDA-GIFs");
			string result;

			// FRAME RATE
			int frameRate = (int) this.numericUpDown_frameRate.Value;

			// RESIZE
			Size? resize = null;
			if ((int) this.numericUpDown_size.Value < Math.Min(this.ImageH.CurrentObject?.Width ?? 0, this.ImageH.CurrentObject?.Height ?? 0))
			{
				resize = new Size((int) this.numericUpDown_size.Value, (int) this.numericUpDown_size.Value);
			}

			// If CTRL down, async call
			if (ModifierKeys == Keys.Control)
			{
				int count = this.Recorder.CachedImages.Count;
				Stopwatch sw = Stopwatch.StartNew();

				this.GuiB.Log("Creating GIF (async) ...", "", 1);
				result = await this.Recorder.CreateGifAsync(folder, this.ImageH.CurrentObject?.Name ?? "animatedGif_", frameRate, true, resize, this.progressBar_load);

				sw.Stop();
				this.GuiB.Log("GIF created (async) ", $"{(sw.ElapsedMilliseconds / count)} ms/F", 1);
			}
			else
			{
				this.GuiB.Log("Creating GIF ...", "", 1);
				result = this.Recorder.CreateGif(folder, this.ImageH.CurrentObject?.Name ?? "animatedGif_", frameRate, true);
			}

			// Msgbox
			if (string.IsNullOrEmpty(result))
			{
				MessageBox.Show("Failed to create GIF", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}
			else
			{
				MessageBox.Show($"GIF created: {result}\n\nSize: {(resize != null ? resize.Value : new Size(this.ImageH.CurrentObject?.Width ?? 0, this.ImageH.CurrentObject?.Height ?? 0))}\nFPS: {frameRate}", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
			}

			// Reset pbar
			this.progressBar_load.Value = 0;

			// Reload image from file
			this.ImageH.CurrentObject?.ResetImage();
		}

		private void checkBox_crosshair_CheckedChanged(object sender, EventArgs e)
		{
			// Toggle crosshair in picturebox
			if (this.checkBox_crosshair.Checked)
			{
				this.ImageH.ShowCrosshair = true;
			}
			else
			{
				this.ImageH.ShowCrosshair = false;
			}

			// Refresh picturebox
			this.ReselectImage();
		}

		private void button_addKernel_Click(object sender, EventArgs e)
		{
			// Call GuiB to open kernel editor (Form)
			this.GuiB.OpenKernelEditor();
		}

		private void comboBox_kernels_SelectedIndexChanged(object sender, EventArgs e)
		{

		}

		private void checkBox_optionalArgsOnly_CheckedChanged(object sender, EventArgs e)
		{
			this.GuiB.BuildPanel(0.55f, this.checkBox_optionalArgsOnly.Checked);
		}

		private async void button_autoFractal_Click(object sender, EventArgs e)
		{
			// Check initialized
			if (this.ContextH.KernelH == null)
			{
				MessageBox.Show("Context not initialized", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			AutoFractalDialogResult? dialogResult = this.GuiB.OpenAutoFractalDialog(this.comboBox_kernels.SelectedItem?.ToString() ?? "");
			if (dialogResult == null)
			{
				return;
			}

			string kernelName = dialogResult.KernelName ?? this.comboBox_kernels.SelectedItem?.ToString() ?? "mandelbrotFullAutoPrecise01";
			Size size = new Size(dialogResult.Width, dialogResult.Height);
			int steps = dialogResult.Steps;
			double initZoom = dialogResult.InitialZoom;
			double zoomIncCoeff = dialogResult.ZoomIncrementCoeff;
			int iterCoeff = dialogResult.IterationCoeff;
			Color col = dialogResult.BaseColor;
			string exportName = dialogResult.ExportFileName;

			// Perform auto fractal -> IntPtr list
			//List<IntPtr> results = this.ContextH.KernelH?.PerformAutoFractal(kernelName, size, steps, initZoom, zoomIncCoeff, iterCoeff, col, this.checkBox_silent.Checked) ?? [];

			List<IntPtr> results = await this.ContextH.KernelH.PerformAutoFractalAsync(kernelName, size, steps, initZoom, zoomIncCoeff, iterCoeff, col, this.progressBar_load, this.checkBox_silent.Checked);
			if (results.Count == 0)
			{
				MessageBox.Show("Failed to create fractal", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			// Create images
			List<ImageObject> objects = [];
			for (int i = 0; i < results.Count; i++)
			{
				byte[] pixels = this.ContextH.MemoryH?.PullData<byte>(results[i], true, this.checkBox_silent.Checked) ?? [];
				ImageObject obj = new(pixels, size);
				obj.Name = $"{kernelName}_{i}";
				objects.Add(obj);
			}

			this.Recorder.CachedImages = objects.Select(x => x.Img ?? new Bitmap(1, 1)).ToList();
			this.Recorder.CountLabel.Text = $"Images: {this.Recorder.CachedImages.Count}";

			// Reset progress bar
			this.progressBar_load.Value = 0;
			this.progressBar_load.Maximum = objects.Count;

			// Create gif
			string result = await this.Recorder.CreateGifAsync(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyPictures), "CUDA-GIFs"), !string.IsNullOrEmpty(exportName) ? exportName : kernelName + "_autoFractal_", dialogResult.Fps, true, new Size(dialogResult.Width, dialogResult.Height), this.progressBar_load);
			if (string.IsNullOrEmpty(result))
			{
				MessageBox.Show("Failed to create GIF", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}
			else
			{
				MessageBox.Show($"GIF created: {result}\n\nSize: {new Size(dialogResult.Width, dialogResult.Height)}\nFPS: {dialogResult.Fps}", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
			}

			// Reset progress bar
			this.progressBar_load.Value = 0;
			this.progressBar_load.Maximum = 100;

			// Fill images listbox
			this.ImageH.FillImagesListBox();
		}

		private void button_fullScreen_Click(object sender, EventArgs e)
		{
			if (!this.MandelbrotMode)
			{
				MessageBox.Show("Fullscreen is only available in Mandelbrot mode", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			if (this.fullScreenForm != null)
			{
				// Bereits aktiv, also schließen
				this.fullScreenForm.Close();
				this.fullScreenForm = null;
				return;
			}

			// Neues Fullscreen-Form erstellen
			this.fullScreenForm = new Form
			{
				FormBorderStyle = FormBorderStyle.None,
				TopMost = true,
				WindowState = FormWindowState.Maximized,
				BackColor = Color.Black,
				StartPosition = FormStartPosition.Manual,
				KeyPreview = true
			};

			// Picturebox Location mittig
			//Point location = new Point((this.fullScreenForm.Width - this.pictureBox_view.Width) / 2, (this.fullScreenForm.Height - this.pictureBox_view.Height) / 2);

			// PictureBox in Originalgröße oder gestreckt anzeigen
			PictureBox pb = new()
			{
				SizeMode = PictureBoxSizeMode.AutoSize,
				Image = this.pictureBox_view.Image,
				BackColor = Color.Black
			};

			// Reuse all existing event handlers
			pb.MouseDown += this.pictureBox_view_MouseDown!;
			pb.MouseMove += this.pictureBox_view_MouseMove!;
			pb.MouseUp += this.pictureBox_view_MouseUp!;
			pb.MouseWheel += this.pictureBox_view_MouseWheel!;
			pb.Paint += this.PictureBox_view_Paint!;
			pb.Focus(); // Damit MouseWheel funktioniert



			this.fullScreenForm.Controls.Add(pb);
			this.ImageH.SetPictureBox(pb);   // auf neue PB umstellen

			// ESC beenden
			this.fullScreenForm.KeyDown += (s, args) =>
			{
				if (args.KeyCode == Keys.Escape)
				{
					this.fullScreenForm?.Close();
					this.fullScreenForm = null;

					this.ImageH.SetPictureBox(this.pictureBox_view); // zurücksetzen
				}
			};


			this.fullScreenForm.Show();
			this.fullScreenForm.Focus(); // wichtig für ESC
		}

	}
}
