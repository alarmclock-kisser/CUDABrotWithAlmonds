
namespace CUDABrotWithAlmonds
{
	public class GuiBuilder
	{
		// ----- ----- ATTRIBUTES ----- ----- \\
		private string Repopath;
		private ListBox LogList;
		private CudaContextHandling ContextH;
		private ImageHandling ImageH;
		private Panel ArgumentsPanel;



		private List<NumericUpDown> NumericsList = [];

		// ----- ----- CONSTRUCTORS ----- ----- \\
		public GuiBuilder(string repopath, ListBox listBox_log, CudaContextHandling contextH, ImageHandling imageH, Panel panel_kernel)
		{
			this.Repopath = repopath;
			this.LogList = listBox_log;
			this.ContextH = contextH;
			this.ImageH = imageH;
			this.ArgumentsPanel = panel_kernel;

			// Register events
			this.ArgumentsPanel.MouseDoubleClick += (s, e) => this.BuildPanel();
		}





		// ----- ----- METHODS ----- ----- \\
		public string Log(string message = "", string inner = "", int indent = 0)
		{
			string indentString = new string(' ', indent);
			string logMessage = $"[Gui] {indentString}{message} ({inner})";
			this.LogList.Items.Add(logMessage);
			this.LogList.TopIndex = this.LogList.Items.Count - 1;
			return logMessage;
		}


		public void BuildPanel(float inputWidthPart = 0.55f)
		{
			// Clear panel & get dimensions
			this.ArgumentsPanel.Controls.Clear();
			this.NumericsList.Clear();
			int maxWidth = this.ArgumentsPanel.Width;
			int maxHeight = this.ArgumentsPanel.Height;
			int inputWidth = (int) (maxWidth * inputWidthPart);

			// Get kernelArgs
			Dictionary<string, Type> arguments = this.ContextH.KernelH?.GetArguments() ?? [];

			// Loop through arguments
			int y = 10;
			foreach (var arg in arguments)
			{
				string argName = arg.Key;
				Type argType = arg.Value;

				// Create label
				Label label = new()
				{
					Name = $"label_arg_{argName}",
					Text = "'" + argName + "'",
					Location = new Point(10, y),
					Size = new Size(maxWidth - 20 - inputWidth, 23)
				};
				this.ArgumentsPanel.Controls.Add(label);

				// Create numeric input
				NumericUpDown numeric = new()
				{
					Name = $"input_arg_{argName}",
					Location = new Point(maxWidth - inputWidth, y),
					Size = new Size(inputWidth - 10, 23),
					Minimum = argType == typeof(char) ? byte.MinValue :
						argType == typeof(IntPtr) ? 0 :
						argType == typeof(sbyte) ? sbyte.MinValue :
						argType == typeof(short) ? short.MinValue :
						argType == typeof(ushort) ? ushort.MinValue :
						argType == typeof(int) ? int.MinValue :
						argType == typeof(uint) ? uint.MinValue :
						argType == typeof(long) ? long.MinValue :
						argType == typeof(ulong) ? ulong.MinValue :
						argType == typeof(float) ? decimal.MinValue :
						argType == typeof(double) ? decimal.MinValue :
						argType == typeof(decimal) ? decimal.MinValue : 0,
					Maximum = argType == typeof(char) ? 255 :
						argType == typeof(IntPtr) ? long.MaxValue :
						argType == typeof(sbyte) ? 127 :
						argType == typeof(short) ? 32767 :
						argType == typeof(ushort) ? 65535 :
						argType == typeof(int) ? int.MaxValue :
						argType == typeof(uint) ? uint.MaxValue :
						argType == typeof(long) ? long.MaxValue :
						argType == typeof(ulong) ? ulong.MaxValue :
						argType == typeof(float) ? decimal.MaxValue :
						argType == typeof(double) ? decimal.MaxValue :
						argType == typeof(decimal) ? decimal.MaxValue : 0,
					Value = argType == typeof(IntPtr) ? (long)(this.ImageH.CurrentObject?.Pointer ?? 0) :
							argName.ToLower().Contains("width") ? this.ImageH.CurrentObject?.Width ?? 0 :
							argName.ToLower().Contains("height") ? this.ImageH.CurrentObject?.Height ?? 0 :
							argName.ToLower().Contains("channels") ? 4 :
							argName.ToLower().Contains("bitdepth") ? this.ImageH.CurrentObject?.BitsPerPixel / 4 ?? 0 : 0,
					DecimalPlaces = argType == typeof(float) ? 4 :
						argType == typeof(double) ? 8 :
						argType == typeof(decimal) ? 12 : 0,
					Increment = argType == typeof(char) ? 1 :
						argType == typeof(sbyte) ? 1 :
						argType == typeof(short) ? 1 :
						argType == typeof(ushort) ? 1 :
						argType == typeof(int) ? 1 :
						argType == typeof(uint) ? 1 :
						argType == typeof(long) ? 1 :
						argType == typeof(ulong) ? 1 :
						argType == typeof(float) ? 0.01m :
						argType == typeof(double) ? 0.0001m :
						argType == typeof(decimal) ? 0.000001m : 0,
				};
				this.ArgumentsPanel.Controls.Add(numeric);
				this.NumericsList.Add(numeric);

				// Add tooltip to numeric
				ToolTip tooltip = new()
				{
					ShowAlways = true,
					AutoPopDelay = 5000,
					InitialDelay = 500,
					ReshowDelay = 500
				};
				tooltip.SetToolTip(numeric, $"{argType.Name}");

				// Special cases for IntPtr (red) and int height/width channels/bitdepth (orange)
				if (argType == typeof(IntPtr))
				{
					numeric.BackColor = Color.Red;
					numeric.Enabled = false;
				}
				else if (argName.ToLower().Contains("width") || argName.ToLower().Contains("height") || argName.ToLower().Contains("channels") || argName.ToLower().Contains("bitdepth"))
				{
					numeric.BackColor = Color.Orange;
					numeric.ReadOnly = true;

					// Register event to toggle enabled on CTRL-click
					numeric.Click += (s, e) =>
					{
						if (Control.ModifierKeys == Keys.Control)
						{
							numeric.ReadOnly = !numeric.ReadOnly;
						}
					};
				}

				// Set location
				y += 30;
			}

			// Add vertical scrollbar if y exceeds panel height
			if (y > maxHeight)
			{
				this.ArgumentsPanel.AutoScroll = true;
				this.ArgumentsPanel.VerticalScroll.Visible = true;
				this.ArgumentsPanel.VerticalScroll.Enabled = true;
				this.ArgumentsPanel.VerticalScroll.Maximum = y - maxHeight + 30;
			}
			else
			{
				this.ArgumentsPanel.AutoScroll = false;
				this.ArgumentsPanel.VerticalScroll.Visible = false;
				this.ArgumentsPanel.VerticalScroll.Enabled = false;
			}

		}

		public object[] GetArgumentValues()
		{
			object[] values = new object[this.NumericsList.Count];
			Type[] types = this.ContextH.KernelH?.GetArguments().Values.ToArray() ?? [];

			for (int i = 0; i < this.NumericsList.Count; i++)
			{
				object value = this.NumericsList[i].Value;
				this.Log(value.ToString() ?? "N/A", types[i].Name, 1);

				// Sonderfall: Zeiger
				if (types[i] == typeof(IntPtr))
				{
					// Statt direktem Cast: erst in long (oder int), dann in IntPtr
					values[i] = new IntPtr(Convert.ToInt64(this.NumericsList[i].Value));
					continue;
				}


				// Sonderfall: Byte (unsigned char)
				if (types[i] == typeof(byte))
				{
					values[i] = Convert.ToByte(value);
					continue;
				}

				// Optional: andere Sonderfälle
				if (types[i] == typeof(sbyte))
				{
					values[i] = Convert.ToSByte(value);
					continue;
				}

				if (types[i] == typeof(char))
				{
					values[i] = Convert.ToByte(value);
					continue;
				}

				// Standardfall: numerisch
				values[i] = Convert.ChangeType(value, types[i]);
			}

			return values;
		}

	}
}