
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
		private List<Label> LabelList = [];

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
			this.LabelList.Clear();
			int maxWidth = this.ArgumentsPanel.Width;
			int maxHeight = this.ArgumentsPanel.Height;
			int inputWidth = (int) (maxWidth * inputWidthPart);

			// Get kernelArgs
			Dictionary<string, Type> arguments = this.ContextH.KernelH?.GetArguments() ?? [];
			var allArgNames = arguments.Keys.ToList();

			// First pass: Create all non-RGB controls
			int y = 10;
			for (int i = 0; i < allArgNames.Count; i++)
			{
				string argName = allArgNames[i];
				Type argType = arguments[argName];

				// Skip if this is part of an RGB triplet (we'll handle them together)
				if (argName.EndsWith("R") && i + 2 < allArgNames.Count &&
					allArgNames[i + 1] == argName.Substring(0, argName.Length - 1) + "G" &&
					allArgNames[i + 2] == argName.Substring(0, argName.Length - 1) + "B")
				{
					continue;
				}
				if (argName.EndsWith("G") || argName.EndsWith("B"))
				{
					continue;
				}

				// Create label
				Label label = new()
				{
					Name = $"label_arg_{argName}",
					Text = argName,
					Location = new Point(10, y),
					Size = new Size(maxWidth - 20 - inputWidth, 23)
				};
				this.ArgumentsPanel.Controls.Add(label);
				this.LabelList.Add(label);

				// Create numeric input
				NumericUpDown numeric = new()
				{
					Name = $"input_arg_{argName}",
					Location = new Point(maxWidth - inputWidth, y),
					Size = new Size(inputWidth - 10, 23),
					Minimum = GetMinimumValue(argType),
					Maximum = GetMaximumValue(argType),
					Value = GetDefaultValue(argName, argType),
					DecimalPlaces = argType == typeof(float) ? 4 :
								  argType == typeof(double) ? 8 :
								  argType == typeof(decimal) ? 12 : 0,
					Increment = GetIncrementValue(argType)
				};

				// Special formatting
				if (argType == typeof(IntPtr))
				{
					numeric.BackColor = Color.LightCoral;
					numeric.Enabled = false;
				}
				else if (IsSpecialParameter(argName))
				{
					numeric.BackColor = Color.LightGoldenrodYellow;
					numeric.ReadOnly = true;
					numeric.Click += (s, e) => {
						if (Control.ModifierKeys == Keys.Control)
						{
							numeric.ReadOnly = !numeric.ReadOnly;
						}
					};
				}

				this.ArgumentsPanel.Controls.Add(numeric);
				this.NumericsList.Add(numeric);
				y += 30;
			}

			// Second pass: Create color pickers for RGB triplets
			for (int i = 0; i < allArgNames.Count - 2; i++)
			{
				string name1 = allArgNames[i];
				string name2 = allArgNames[i + 1];
				string name3 = allArgNames[i + 2];

				if (name1.EndsWith("R") && name2.EndsWith("G") && name3.EndsWith("B") &&
					name1.Substring(0, name1.Length - 1) == name2.Substring(0, name2.Length - 1) &&
					name2.Substring(0, name2.Length - 1) == name3.Substring(0, name3.Length - 1))
				{
					string colorName = name1.Substring(0, name1.Length - 1);

					// Create color picker button
					Button colorButton = new()
					{
						Name = $"button_arg_{colorName}",
						Text = $"{colorName} Color",
						Location = new Point(maxWidth - inputWidth, y),
						Size = new Size(inputWidth - 10, 23),
						BackColor = Color.FromArgb(
							(int) GetDefaultValue(name1, typeof(int)),
							(int) GetDefaultValue(name2, typeof(int)),
							(int) GetDefaultValue(name3, typeof(int))),
						Tag = new string[] { name1, name2, name3 } // Store component names
					};
					UpdateButtonTextColor(colorButton);

					colorButton.Click += (s, e) => {
						ColorDialog cd = new() { Color = colorButton.BackColor };
						if (cd.ShowDialog() == DialogResult.OK)
						{
							colorButton.BackColor = cd.Color;
							UpdateButtonTextColor(colorButton);
						}
					};

					// Create label
					Label colorLabel = new()
					{
						Name = $"label_arg_{colorName}",
						Text = colorName,
						Location = new Point(10, y),
						Size = new Size(maxWidth - 20 - inputWidth, 23)
					};

					this.ArgumentsPanel.Controls.Add(colorLabel);
					this.ArgumentsPanel.Controls.Add(colorButton);
					this.LabelList.Add(colorLabel);
					y += 30;

					// Skip next two elements (G and B)
					i += 2;
				}
			}

			// Add scrollbar if needed
			this.ArgumentsPanel.AutoScroll = y > maxHeight;
		}

		public object[] GetArgumentValues()
		{
			var argsDefinitions = this.ContextH.KernelH?.GetArguments() ?? [];
			object[] values = new object[argsDefinitions.Count];
			var allArgNames = argsDefinitions.Keys.ToList();

			for (int i = 0; i < allArgNames.Count; i++)
			{
				string argName = allArgNames[i];
				Type argType = argsDefinitions[argName];

				// Handle RGB components
				if (argName.EndsWith("R") && i + 2 < allArgNames.Count &&
					allArgNames[i + 1].EndsWith("G") && allArgNames[i + 2].EndsWith("B") &&
					argName.Substring(0, argName.Length - 1) ==
					allArgNames[i + 1].Substring(0, allArgNames[i + 1].Length - 1))
				{
					string buttonName = $"button_arg_{argName.Substring(0, argName.Length - 1)}";
					var button = this.ArgumentsPanel.Controls.OfType<Button>()
								   .FirstOrDefault(b => b.Name == buttonName);

					if (button != null)
					{
						Color c = button.BackColor;
						values[i] = c.R;
						values[i + 1] = c.G;
						values[i + 2] = c.B;
						i += 2; // Skip G and B components
						continue;
					}
				}

				// Handle normal numeric inputs
				var numeric = this.NumericsList.FirstOrDefault(n =>
					n.Name == $"input_arg_{argName}");

				if (numeric != null)
				{
					if (argType == typeof(IntPtr))
					{
						values[i] = new IntPtr(Convert.ToInt64(numeric.Value));
					}
					else if (argType == typeof(char))
					{
						values[i] = Convert.ToChar(Convert.ToInt32(numeric.Value));
					}
					else if (argType == typeof(byte))
					{
						values[i] = Convert.ToByte(numeric.Value);
					}
					else if (argType == typeof(sbyte))
					{
						values[i] = Convert.ToSByte(numeric.Value);
					}
					else if (argType == typeof(short))
					{
						values[i] = Convert.ToInt16(numeric.Value);
					}
					else if (argType == typeof(ushort))
					{
						values[i] = Convert.ToUInt16(numeric.Value);
					}
					else if (argType == typeof(int))
					{
						values[i] = Convert.ToInt32(numeric.Value);
					}
					else if (argType == typeof(uint))
					{
						values[i] = Convert.ToUInt32(numeric.Value);
					}
					else if (argType == typeof(long))
					{
						values[i] = Convert.ToInt64(numeric.Value);
					}
					else if (argType == typeof(ulong))
					{
						values[i] = Convert.ToUInt64(numeric.Value);
					}
					else if (argType == typeof(float))
					{
						values[i] = Convert.ToSingle(numeric.Value);
					}
					else if (argType == typeof(double))
					{
						values[i] = Convert.ToDouble(numeric.Value);
					}
					else if (argType == typeof(decimal))
					{
						values[i] = numeric.Value;
					}
					else
					{
						values[i] = Convert.ChangeType(numeric.Value, argType);
					}
				}
			}

			return values;
		}
		// Helper methods
		private decimal GetMinimumValue(Type type) => type == typeof(char) ? byte.MinValue :
				   type == typeof(sbyte) ? sbyte.MinValue :
				   type == typeof(short) ? short.MinValue :
				   type == typeof(ushort) ? ushort.MinValue :
				   type == typeof(int) ? int.MinValue :
				   type == typeof(uint) ? uint.MinValue :
				   type == typeof(long) ? long.MinValue :
				   type == typeof(ulong) ? ulong.MinValue :
				   type == typeof(float) ? decimal.MinValue :
				   type == typeof(double) ? decimal.MinValue :
				   type == typeof(decimal) ? decimal.MinValue : 0;

		private decimal GetMaximumValue(Type type) => type == typeof(char) ? byte.MaxValue :
				   type == typeof(sbyte) ? sbyte.MaxValue :
				   type == typeof(short) ? short.MaxValue :
				   type == typeof(ushort) ? ushort.MaxValue :
				   type == typeof(int) ? int.MaxValue :
				   type == typeof(uint) ? uint.MaxValue :
				   type == typeof(long) ? long.MaxValue :
				   type == typeof(ulong) ? ulong.MaxValue :
				   type == typeof(float) ? decimal.MaxValue :
				   type == typeof(double) ? decimal.MaxValue :
				   type == typeof(decimal) ? decimal.MaxValue : 0;

		private decimal GetDefaultValue(string argName, Type argType)
		{
			if (argType == typeof(IntPtr))
			{
				return (long) (this.ImageH.CurrentObject?.Pointer ?? IntPtr.Zero);
			}

			if (argName.Contains("width", StringComparison.OrdinalIgnoreCase))
			{
				return this.ImageH.CurrentObject?.Width ?? 0;
			}

			if (argName.Contains("height", StringComparison.OrdinalIgnoreCase))
			{
				return this.ImageH.CurrentObject?.Height ?? 0;
			}

			if (argName.Contains("channel", StringComparison.OrdinalIgnoreCase))
			{
				return 4;
			}

			return argName.Contains("bit", StringComparison.OrdinalIgnoreCase) ?  this.ImageH.CurrentObject?.BitsPerPixel / 8 ?? 0 :  0;
		}

		private decimal GetIncrementValue(Type type) => type == typeof(float) ? 0.01m :
				   type == typeof(double) ? 0.0001m :
				   type == typeof(decimal) ? 0.000001m : 1;

		private bool IsSpecialParameter(string argName) => argName.Contains("width", StringComparison.OrdinalIgnoreCase) ||
				   argName.Contains("height", StringComparison.OrdinalIgnoreCase) ||
				   argName.Contains("channel", StringComparison.OrdinalIgnoreCase) ||
				   argName.Contains("bit", StringComparison.OrdinalIgnoreCase);

		private void UpdateButtonTextColor(Button button) => button.ForeColor = button.BackColor.GetBrightness() < 0.5 ? Color.White : Color.Black;


	}
}