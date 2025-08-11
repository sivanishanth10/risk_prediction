from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import io
from PIL import Image as PILImage

def create_risk_chart(risk_score, risk_level):
    """Create a simple risk visualization chart"""
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Create gauge-like visualization
        colors_list = ['green', 'yellow', 'red']
        if risk_score < 0.3:
            color = colors_list[0]
        elif risk_score < 0.7:
            color = colors_list[1]
        else:
            color = colors_list[2]
        
        # Create horizontal bar
        bars = ax.barh(['Risk Score'], [risk_score], color=color, height=0.3)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Risk Probability')
        ax.set_title(f'Risk Assessment: {risk_level}')
        
        # Add value label
        ax.text(risk_score + 0.02, 0, f'{risk_score:.3f}', va='center', fontsize=12, fontweight='bold')
        
        # Add threshold lines
        ax.axvline(x=0.3, color='green', linestyle='--', alpha=0.7, label='Low Risk Threshold')
        ax.axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='High Risk Threshold')
        
        plt.tight_layout()
        
        # Save to bytes buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
    except Exception as e:
        print(f"Error creating risk chart: {e}")
        return None

def create_risk_factors_chart(risk_factors):
    """Create a risk factors visualization chart"""
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        factors = list(risk_factors.keys())
        values = list(risk_factors.values())
        
        bars = ax.barh(factors, values, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Color code bars based on risk level
        for i, (bar, value) in enumerate(zip(bars, values)):
            if value > 0.7:
                bar.set_color('red')
            elif value > 0.4:
                bar.set_color('orange')
            else:
                bar.set_color('green')
        
        ax.set_xlim(0, 1)
        ax.set_xlabel('Risk Factor Score')
        ax.set_title('Individual Risk Factors Analysis')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        # Save to bytes buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
    except Exception as e:
        print(f"Error creating risk factors chart: {e}")
        return None

def generate_pdf_report(name, prediction, input_data=None, risk_factors=None, shap_img="reports/shap_summary.png"):
    """Generate elaborate PDF report for borrower risk assessment"""
    try:
        # Create reports directory if it doesn't exist
        os.makedirs("reports/borrower_reports", exist_ok=True)
        
        file_path = f"reports/borrower_reports/{name}_credit_risk_report.pdf"
        
        # Create PDF document
        doc = SimpleDocTemplate(file_path, pagesize=A4, 
                              rightMargin=72, leftMargin=72, 
                              topMargin=72, bottomMargin=72)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkblue
        )
        
        normal_style = styles['Normal']
        normal_style.fontSize = 11
        normal_style.spaceAfter = 6
        
        # Build story
        story = []
        
        # Title page
        story.append(Paragraph("Credit Risk Assessment Report", title_style))
        story.append(Spacer(1, 20))
        
        # Report metadata
        current_date = datetime.now().strftime("%B %d, %Y")
        story.append(Paragraph(f"<b>Report Date:</b> {current_date}", normal_style))
        story.append(Paragraph(f"<b>Borrower Name:</b> {name}", normal_style))
        story.append(Paragraph(f"<b>Report ID:</b> CR-{datetime.now().strftime('%Y%m%d%H%M%S')}", normal_style))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))
        
        risk_score = float(prediction)
        if risk_score < 0.3:
            risk_level = "Low Risk"
            recommendation = "APPROVED"
            risk_color = "green"
            summary_text = f"The borrower demonstrates a strong credit profile with a risk score of {risk_score:.3f}. The application is recommended for approval with standard terms."
        elif risk_score < 0.7:
            risk_level = "Medium Risk"
            recommendation = "REVIEW REQUIRED"
            risk_color = "orange"
            summary_text = f"The borrower presents a moderate credit risk with a score of {risk_score:.3f}. Additional documentation and review are recommended before making a decision."
        else:
            risk_level = "High Risk"
            recommendation = "NOT RECOMMENDED"
            risk_color = "red"
            summary_text = f"The borrower exhibits significant credit risk with a score of {risk_score:.3f}. The application is not recommended for approval at this time."
        
        story.append(Paragraph(f"<b>Overall Risk Assessment:</b> {risk_level}", normal_style))
        story.append(Paragraph(f"<b>Risk Score:</b> {risk_score:.3f}", normal_style))
        story.append(Paragraph(f"<b>Recommendation:</b> {recommendation}", normal_style))
        story.append(Paragraph(summary_text, normal_style))
        story.append(Spacer(1, 20))
        
        # Risk Visualization
        story.append(Paragraph("Risk Assessment Visualization", heading_style))
        
        # Create and add risk chart
        risk_chart_buffer = create_risk_chart(risk_score, risk_level)
        if risk_chart_buffer:
            try:
                risk_chart_buffer.seek(0)
                risk_img = Image(risk_chart_buffer, width=5*inch, height=3*inch)
                story.append(risk_img)
                story.append(Spacer(1, 12))
            except Exception as e:
                story.append(Paragraph(f"Risk chart could not be generated: {str(e)}", normal_style))
        
        # Input Data Summary
        if input_data:
            story.append(Paragraph("Borrower Information", heading_style))
            
            # Create table for input data
            input_table_data = [['Field', 'Value']]
            for key, value in input_data.items():
                # Format field names for better readability
                formatted_key = key.replace('_', ' ').title()
                if isinstance(value, float):
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)
                input_table_data.append([formatted_key, formatted_value])
            
            input_table = Table(input_table_data, colWidths=[3*inch, 2*inch])
            input_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(input_table)
            story.append(Spacer(1, 20))
        
        # Risk Factors Analysis
        if risk_factors:
            story.append(Paragraph("Risk Factors Analysis", heading_style))
            
            # Create and add risk factors chart
            risk_factors_chart_buffer = create_risk_factors_chart(risk_factors)
            if risk_factors_chart_buffer:
                try:
                    risk_factors_chart_buffer.seek(0)
                    risk_factors_img = Image(risk_factors_chart_buffer, width=6*inch, height=4*inch)
                    story.append(risk_factors_img)
                    story.append(Spacer(1, 12))
                except Exception as e:
                    story.append(Paragraph(f"Risk factors chart could not be generated: {str(e)}", normal_style))
            
            # Add risk factors table
            risk_table_data = [['Risk Factor', 'Score', 'Risk Level']]
            for factor, score in risk_factors.items():
                if score > 0.7:
                    level = "High"
                    color = colors.red
                elif score > 0.4:
                    level = "Medium"
                    color = colors.orange
                else:
                    level = "Low"
                    color = colors.green
                
                risk_table_data.append([factor, f"{score:.3f}", level])
            
            risk_table = Table(risk_table_data, colWidths=[2.5*inch, 1.5*inch, 1*inch])
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            story.append(risk_table)
            story.append(Spacer(1, 20))
        
        # SHAP Analysis
        story.append(Paragraph("Feature Importance Analysis (SHAP)", heading_style))
        if os.path.exists(shap_img):
            try:
                shap_img_obj = Image(shap_img, width=5*inch, height=4*inch)
                story.append(shap_img_obj)
                story.append(Paragraph("This SHAP (SHapley Additive exPlanations) plot shows the contribution of each feature to the final prediction.", normal_style))
            except Exception as e:
                story.append(Paragraph(f"SHAP visualization could not be included: {str(e)}", normal_style))
        else:
            story.append(Paragraph("SHAP visualization not available for this analysis.", normal_style))
        
        story.append(Spacer(1, 20))
        
        # Recommendations
        story.append(Paragraph("Recommendations", heading_style))
        
        if risk_score < 0.3:
            recommendations = [
                "Maintain current credit management practices",
                "Consider offering premium terms and conditions",
                "Monitor credit utilization to maintain low risk status",
                "Regular credit reviews every 6 months"
            ]
        elif risk_score < 0.7:
            recommendations = [
                "Request additional financial documentation",
                "Consider secured lending options",
                "Implement stricter monitoring and reporting",
                "Review application after 3-6 months",
                "Consider co-signer or collateral requirements"
            ]
        else:
            recommendations = [
                "Application not recommended for approval",
                "Consider alternative lending products if available",
                "Recommend credit counseling services",
                "Re-evaluate after significant credit improvement",
                "Consider secured or specialized lending programs"
            ]
        
        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", normal_style))
        
        story.append(Spacer(1, 20))
        
        # Risk Mitigation Strategies
        story.append(Paragraph("Risk Mitigation Strategies", heading_style))
        
        mitigation_text = """
        • <b>Credit Monitoring:</b> Implement regular credit score monitoring and reporting
        • <b>Payment Reminders:</b> Set up automated payment reminders and notifications
        • <b>Financial Education:</b> Provide resources for financial literacy and credit management
        • <b>Regular Reviews:</b> Schedule periodic portfolio reviews and risk assessments
        • <b>Documentation:</b> Maintain comprehensive records of all lending decisions and communications
        """
        
        story.append(Paragraph(mitigation_text, normal_style))
        story.append(Spacer(1, 20))
        
        # Footer
        story.append(Paragraph("---", normal_style))
        story.append(Paragraph("Generated by Credit Risk Prediction System", normal_style))
        story.append(Paragraph(f"Report generated on: {current_date}", normal_style))
        story.append(Paragraph("This report is confidential and intended for internal use only.", normal_style))
        
        # Build PDF
        doc.build(story)
        
        return file_path
        
    except Exception as e:
        print(f"Error generating PDF report: {e}")
        return None

def create_sample_pdf_report():
    """Create a sample PDF report for demonstration purposes"""
    try:
        # Sample data
        sample_input_data = {
            'RevolvingUtilizationOfUnsecuredLines': 45.2,
            'age': 32,
            'NumberOfTime30-59DaysPastDueNotWorse': 1,
            'DebtRatio': 0.35,
            'MonthlyIncome': 6500,
            'NumberOfOpenCreditLinesAndLoans': 8,
            'NumberOfTimes90DaysLate': 0,
            'NumberRealEstateLoansOrLines': 1,
            'NumberOfTime60-89DaysPastDueNotWorse': 0,
            'NumberOfDependents': 2
        }
        
        sample_risk_factors = {
            'Revolving Utilization': 0.452,
            'Age Factor': 0.086,
            'Payment History': 0.003,
            'Debt Burden': 0.35,
            'Income Stability': 0.03
        }
        
        # Generate sample report
        return generate_pdf_report(
            name="Sample Borrower",
            prediction="0.425",
            input_data=sample_input_data,
            risk_factors=sample_risk_factors
        )
    except Exception as e:
        print(f"Error creating sample PDF report: {e}")
        return None
