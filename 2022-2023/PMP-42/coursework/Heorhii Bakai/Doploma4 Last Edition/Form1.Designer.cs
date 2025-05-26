namespace Doploma4_Last_Edition
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea3 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
            System.Windows.Forms.DataVisualization.Charting.Legend legend3 = new System.Windows.Forms.DataVisualization.Charting.Legend();
            System.Windows.Forms.DataVisualization.Charting.Series series15 = new System.Windows.Forms.DataVisualization.Charting.Series();
            System.Windows.Forms.DataVisualization.Charting.Series series16 = new System.Windows.Forms.DataVisualization.Charting.Series();
            System.Windows.Forms.DataVisualization.Charting.Series series17 = new System.Windows.Forms.DataVisualization.Charting.Series();
            System.Windows.Forms.DataVisualization.Charting.Series series18 = new System.Windows.Forms.DataVisualization.Charting.Series();
            System.Windows.Forms.DataVisualization.Charting.Series series19 = new System.Windows.Forms.DataVisualization.Charting.Series();
            System.Windows.Forms.DataVisualization.Charting.Series series20 = new System.Windows.Forms.DataVisualization.Charting.Series();
            System.Windows.Forms.DataVisualization.Charting.Series series21 = new System.Windows.Forms.DataVisualization.Charting.Series();
            this.Chart1 = new System.Windows.Forms.DataVisualization.Charting.Chart();
            this.button1 = new System.Windows.Forms.Button();
            this.PeopleCount = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.InfectiousCount = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.Beta = new System.Windows.Forms.TextBox();
            this.Alpha = new System.Windows.Forms.TextBox();
            this.Gamma = new System.Windows.Forms.TextBox();
            this.Mu = new System.Windows.Forms.TextBox();
            this.Theta = new System.Windows.Forms.TextBox();
            this.Phi = new System.Windows.Forms.TextBox();
            this.P = new System.Windows.Forms.TextBox();
            this.Q = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.label5 = new System.Windows.Forms.Label();
            this.label6 = new System.Windows.Forms.Label();
            this.label7 = new System.Windows.Forms.Label();
            this.label8 = new System.Windows.Forms.Label();
            this.label9 = new System.Windows.Forms.Label();
            this.label10 = new System.Windows.Forms.Label();
            this.label11 = new System.Windows.Forms.Label();
            this.listBox1 = new System.Windows.Forms.ListBox();
            ((System.ComponentModel.ISupportInitialize)(this.Chart1)).BeginInit();
            this.SuspendLayout();
            // 
            // Chart1
            // 
            chartArea3.Name = "ChartArea1";
            this.Chart1.ChartAreas.Add(chartArea3);
            this.Chart1.Cursor = System.Windows.Forms.Cursors.Default;
            legend3.Name = "Legend1";
            this.Chart1.Legends.Add(legend3);
            this.Chart1.Location = new System.Drawing.Point(28, 21);
            this.Chart1.Name = "Chart1";
            this.Chart1.RightToLeft = System.Windows.Forms.RightToLeft.No;
            series15.ChartArea = "ChartArea1";
            series15.ChartType = System.Windows.Forms.DataVisualization.Charting.SeriesChartType.Line;
            series15.LabelBorderWidth = 2;
            series15.Legend = "Legend1";
            series15.Name = "Susceptible";
            series16.ChartArea = "ChartArea1";
            series16.ChartType = System.Windows.Forms.DataVisualization.Charting.SeriesChartType.Line;
            series16.LabelBorderWidth = 2;
            series16.Legend = "Legend1";
            series16.Name = "Exposed";
            series17.ChartArea = "ChartArea1";
            series17.ChartType = System.Windows.Forms.DataVisualization.Charting.SeriesChartType.Line;
            series17.LabelBorderWidth = 2;
            series17.Legend = "Legend1";
            series17.Name = "Infection";
            series18.ChartArea = "ChartArea1";
            series18.ChartType = System.Windows.Forms.DataVisualization.Charting.SeriesChartType.Line;
            series18.LabelBorderWidth = 2;
            series18.Legend = "Legend1";
            series18.Name = "Hospitalized";
            series19.ChartArea = "ChartArea1";
            series19.ChartType = System.Windows.Forms.DataVisualization.Charting.SeriesChartType.Line;
            series19.LabelBorderWidth = 2;
            series19.Legend = "Legend1";
            series19.Name = "Critical";
            series20.ChartArea = "ChartArea1";
            series20.ChartType = System.Windows.Forms.DataVisualization.Charting.SeriesChartType.Line;
            series20.LabelBorderWidth = 2;
            series20.Legend = "Legend1";
            series20.Name = "Recovered";
            series21.ChartArea = "ChartArea1";
            series21.ChartType = System.Windows.Forms.DataVisualization.Charting.SeriesChartType.Line;
            series21.LabelBorderWidth = 2;
            series21.Legend = "Legend1";
            series21.Name = "Deceased";
            this.Chart1.Series.Add(series15);
            this.Chart1.Series.Add(series16);
            this.Chart1.Series.Add(series17);
            this.Chart1.Series.Add(series18);
            this.Chart1.Series.Add(series19);
            this.Chart1.Series.Add(series20);
            this.Chart1.Series.Add(series21);
            this.Chart1.Size = new System.Drawing.Size(829, 390);
            this.Chart1.TabIndex = 0;
            this.Chart1.Text = "chart1";
            this.Chart1.Click += new System.EventHandler(this.chart1_Click);
            // 
            // button1
            // 
            this.button1.Location = new System.Drawing.Point(11, 417);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(147, 57);
            this.button1.TabIndex = 1;
            this.button1.Text = "Start";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // PeopleCount
            // 
            this.PeopleCount.Location = new System.Drawing.Point(175, 436);
            this.PeopleCount.Name = "PeopleCount";
            this.PeopleCount.Size = new System.Drawing.Size(190, 20);
            this.PeopleCount.TabIndex = 2;
            this.PeopleCount.Text = "10000";
            this.PeopleCount.TextChanged += new System.EventHandler(this.PeopleCount_TextChanged);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(175, 417);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(160, 13);
            this.label1.TabIndex = 3;
            this.label1.Text = "Загальна кількість населення";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(178, 476);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(115, 13);
            this.label2.TabIndex = 4;
            this.label2.Text = "Кількість інфікованих";
            // 
            // InfectiousCount
            // 
            this.InfectiousCount.Location = new System.Drawing.Point(175, 493);
            this.InfectiousCount.Name = "InfectiousCount";
            this.InfectiousCount.Size = new System.Drawing.Size(190, 20);
            this.InfectiousCount.TabIndex = 5;
            this.InfectiousCount.Text = "5";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(396, 417);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(67, 13);
            this.label3.TabIndex = 6;
            this.label3.Text = "Коефіцієнти";
            // 
            // Beta
            // 
            this.Beta.Location = new System.Drawing.Point(399, 454);
            this.Beta.Name = "Beta";
            this.Beta.Size = new System.Drawing.Size(71, 20);
            this.Beta.TabIndex = 10;
            this.Beta.Text = "0,4";
            // 
            // Alpha
            // 
            this.Alpha.Location = new System.Drawing.Point(399, 498);
            this.Alpha.Name = "Alpha";
            this.Alpha.Size = new System.Drawing.Size(71, 20);
            this.Alpha.TabIndex = 11;
            this.Alpha.Text = "0,3";
            // 
            // Gamma
            // 
            this.Gamma.Location = new System.Drawing.Point(399, 544);
            this.Gamma.Name = "Gamma";
            this.Gamma.Size = new System.Drawing.Size(71, 20);
            this.Gamma.TabIndex = 12;
            this.Gamma.Text = "0,2";
            this.Gamma.TextChanged += new System.EventHandler(this.textBox3_TextChanged);
            // 
            // Mu
            // 
            this.Mu.Location = new System.Drawing.Point(399, 592);
            this.Mu.Name = "Mu";
            this.Mu.Size = new System.Drawing.Size(71, 20);
            this.Mu.TabIndex = 13;
            this.Mu.Text = "0,05";
            // 
            // Theta
            // 
            this.Theta.Location = new System.Drawing.Point(640, 454);
            this.Theta.Name = "Theta";
            this.Theta.Size = new System.Drawing.Size(71, 20);
            this.Theta.TabIndex = 14;
            this.Theta.Text = "0,4";
            // 
            // Phi
            // 
            this.Phi.Location = new System.Drawing.Point(640, 498);
            this.Phi.Name = "Phi";
            this.Phi.Size = new System.Drawing.Size(71, 20);
            this.Phi.TabIndex = 15;
            this.Phi.Text = "0,3";
            // 
            // P
            // 
            this.P.Location = new System.Drawing.Point(640, 545);
            this.P.Name = "P";
            this.P.Size = new System.Drawing.Size(71, 20);
            this.P.TabIndex = 16;
            this.P.Text = "0,6";
            // 
            // Q
            // 
            this.Q.Location = new System.Drawing.Point(640, 592);
            this.Q.Name = "Q";
            this.Q.Size = new System.Drawing.Size(71, 20);
            this.Q.TabIndex = 17;
            this.Q.Text = "0,3";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(396, 436);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(85, 13);
            this.label4.TabIndex = 18;
            this.label4.Text = "Шанс захворіти";
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(396, 482);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(234, 13);
            this.label5.TabIndex = 19;
            this.label5.Text = "Шанс перейти з прихованої фази до хвороби";
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(396, 528);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(79, 13);
            this.label6.TabIndex = 20;
            this.label6.Text = "Шанс одужати";
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(396, 574);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(70, 13);
            this.label7.TabIndex = 21;
            this.label7.Text = "Шанс смерті";
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(637, 437);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(104, 13);
            this.label8.TabIndex = 22;
            this.label8.Text = "Шанс госпіталізації";
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(637, 482);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(125, 13);
            this.label9.TabIndex = 23;
            this.label9.Text = "Шанс  критичних станів";
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.Location = new System.Drawing.Point(637, 518);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(202, 26);
            this.label10.TabIndex = 24;
            this.label10.Text = "Шанс  заражених, які не потрапляють \r\nдо лікарні";
            // 
            // label11
            // 
            this.label11.AutoSize = true;
            this.label11.Location = new System.Drawing.Point(637, 565);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(171, 26);
            this.label11.TabIndex = 25;
            this.label11.Text = "Шанс  госпіталізованих, \r\nякі переходять у критичний стан";
            // 
            // listBox1
            // 
            this.listBox1.FormattingEnabled = true;
            this.listBox1.Location = new System.Drawing.Point(11, 518);
            this.listBox1.Name = "listBox1";
            this.listBox1.Size = new System.Drawing.Size(354, 95);
            this.listBox1.TabIndex = 26;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(852, 618);
            this.Controls.Add(this.listBox1);
            this.Controls.Add(this.label11);
            this.Controls.Add(this.label10);
            this.Controls.Add(this.label9);
            this.Controls.Add(this.label8);
            this.Controls.Add(this.label7);
            this.Controls.Add(this.label6);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.Q);
            this.Controls.Add(this.P);
            this.Controls.Add(this.Phi);
            this.Controls.Add(this.Theta);
            this.Controls.Add(this.Mu);
            this.Controls.Add(this.Gamma);
            this.Controls.Add(this.Alpha);
            this.Controls.Add(this.Beta);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.InfectiousCount);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.PeopleCount);
            this.Controls.Add(this.button1);
            this.Controls.Add(this.Chart1);
            this.Name = "Form1";
            this.Text = "Form1";
            ((System.ComponentModel.ISupportInitialize)(this.Chart1)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion
        private System.Windows.Forms.Button button1;
        public System.Windows.Forms.DataVisualization.Charting.Chart Chart1;
        private System.Windows.Forms.TextBox PeopleCount;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox InfectiousCount;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox Beta;
        private System.Windows.Forms.TextBox Alpha;
        private System.Windows.Forms.TextBox Gamma;
        private System.Windows.Forms.TextBox Mu;
        private System.Windows.Forms.TextBox Theta;
        private System.Windows.Forms.TextBox Phi;
        private System.Windows.Forms.TextBox P;
        private System.Windows.Forms.TextBox Q;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.Label label10;
        private System.Windows.Forms.Label label11;
        private System.Windows.Forms.ListBox listBox1;
    }
}

