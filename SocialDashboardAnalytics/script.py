import os
import subprocess
import numpy as np
import pafy
import requests 
import re
import csv
import urllib2
import xlsxwriter

class ScriptBuilding:
	
	def youtubeFeature(self):
		url=self.url
		self.col1=0
		v = pafy.new(url)
		self.worksheet.write(self.row1, self.col1,v.title)
		self.col1 += 1
		self.worksheet.write(self.row1, self.col1, v.duration)
		self.col1 += 1
		self.worksheet.write(self.row1, self.col1, v.rating)
		self.col1 += 1
		self.worksheet.write(self.row1, self.col1, v.likes)
		self.col1 += 1
		self.worksheet.write(self.row1, self.col1, v.dislikes)
		self.col1 += 1
		self.worksheet.write(self.row1, self.col1, v.author)
		self.col1 += 1
		self.worksheet.write(self.row1, self.col1, v.length)
		self.col1 += 1
		self.worksheet.write(self.row1, self.col1, v.viewcount)
		self.col1 += 1
		self.worksheet.write(self.row1, self.col1, v.thumb)
		self.col1 += 1
		self.worksheet.write(self.row1, self.col1, v.videoid)
    		self.col1 += 1
		self.worksheet.write(self.row1, self.col1, v.published)
    		self.col1 += 1
		self.worksheet.write(self.row1, self.col1, v.category)
    		self.col1 += 1
		self.worksheet.write(self.row1, self.col1, v.description)
    		self.row1 += 1
		print('title',v.title)
		print('duration',v.duration)
		print('rating',v.rating)
		print('author',v.author)
		print('length',v.length)
		print('keywords',v.keywords)
		print('thumb',v.thumb)
		print('vidoeid',v.videoid)
		print('viewcount',v.viewcount)
		print('likes',v.likes)
		print('dislikes',v.dislikes)	
		print('published: ',v.published)	
		print('category: ',v.category)	
		print('description: ',v.description)	
		
	def processing(self):
		features=[]
		# Create a workbook and add a worksheet.
		self.workbook = xlsxwriter.Workbook('YoutubeData_new.xlsx')
		self.worksheet = self.workbook.add_worksheet()
		self.row1 = 0
		self.col1 = 0
		#read csv file which contain name of file and its relative youtube link
		with open("videofiles.csv", 'rb') as f:
			reader = csv.reader(f)
			headers = reader.next()
			column = {h:[] for h in headers}
			#reading csv file
			for row in reader:
				#fileName=row[0]
				url=row[0]
				#print(fileName)
				print(url)
				if(url!=""):
					#self.feature=[]
					#self.fileName=fileName
					self.url=url
					#calling function to fetch yourube metadata
					self.youtubeFeature()
					#calling function to fetch audio features
					#self.audioFeature()
					#deleting directory
					#_notused=subprocess.check_output(["rm -r "+fileName],shell=True)			
					#print ('Feature array:-', self.feature)
					
		
#creating object for class which contain Scipt for building feature vector
obj=ScriptBuilding()

#calling script for extracting features
obj.processing()
