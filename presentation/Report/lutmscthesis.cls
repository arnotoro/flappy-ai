% $Id: lutmscthesis.cls,v 1.21 2010/10/08 13:39:40 paalanen Exp $

% This document style is originally extracted from
% P. Paalanen. Real-time Imaging and Mosaicing of Planar Surfaces.
% Master's thesis, Lappeenranta University of Technology. October 2006.
% 3.11.2006 - pq
% 3.10.2017 - ak

% Prehistory:
% - unknown, original version
% - Jarmo "trewas" Ilonen, masters thesis, 2003
% - Pekka "PQ" Paalanen, information processing project, 2004,
%     hints about graphicx and making PDF from Pasi Valminen

% Modifications:
% - Pekka Paalanen, 2010 (modified and cleaned up for TexLive 2008 and pdf)
% - (your name, work, year)

% Problems:
% - REFERENCES link goes to wrong page

% Packages that this document class includes automatically:
% geometry, graphicx, subfig, caption(3)

% See the accompanying thesis.tex for usage information,
% and Makefile for compiling the document.


% TODO: acronyms

% ------------ Identification
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{lutmscthesis}[%
2017/10/03 LUT Master's Thesis $Revision: 1.22 $]


% ------------ Initial code
\RequirePackage{ifthen}
\RequirePackage{calc}

\newif\if@hyperref@enabled

% ------------ Declaration of options


% ------------ Execution of options
\ProcessOptions \relax


% ------------ Package loading
\LoadClass[a4paper,11pt]{article}[2001/04/21]

\RequirePackage{geometry}
\RequirePackage{graphicx}[1999/02/16]
\RequirePackage{extramarks}
\RequirePackage{placeins}

% the caption3 package
\RequirePackage{caption}[2006/01/12]
\RequirePackage{subfig}


% ------------ Main code

\AtBeginDocument{%
  \usepackage{lastpage}%
  \@ifpackageloaded{hyperref}{%
    \@hyperref@enabledtrue%
    \hypersetup{pdfauthor={\@author},pdftitle={\@title},pdfsubtitle={\@subtitle},%
      pdfkeywords={\EmitKeywords}}%
  }{%
    \@hyperref@enabledfalse%
  }%
}

\newcommand\MakeStringVar[2][\relax]{%
  \ifx#1\relax%
    \expandafter\newcommand\csname Emit#2\endcsname{%
      {\scriptsize (Use {\tt\textbackslash #2} to replace this text.)}}%
  \else%
    \expandafter\newcommand\csname Emit#2\endcsname{#1}%
  \fi%
  \expandafter\newcommand\csname #2\endcsname[1]{%
     \expandafter\renewcommand\csname Emit#2\endcsname{##1}%
  }%
}

\newcommand\MakePeopleVar[3]{%
  \expandafter\newcommand\csname Emit#1\endcsname{#3}%
  \expandafter\newcommand\csname #1Title\endcsname{#1}%
  \expandafter\newcommand\csname #1Titlefin\endcsname{#2}%
  \expandafter\newcommand\csname #1\endcsname[1]{%
     \expandafter\renewcommand\csname Emit#1\endcsname{##1}%
     \expandafter\renewcommand\csname #1Title\endcsname{#1}%
     \expandafter\renewcommand\csname #1Titlefin\endcsname{#2}%
  }%
  \expandafter\newcommand\csname #1s\endcsname[1]{%
     \expandafter\renewcommand\csname Emit#1\endcsname{##1}%
     \expandafter\renewcommand\csname #1Title\endcsname{#1s}%
     \expandafter\renewcommand\csname #1Titlefin\endcsname{#2t}%
  }%
}

% constants
\newcommand\school{Lappeenranta-Lahti University of Technology LUT}

\MakeStringVar[School of Engineering Science]{School}
\MakeStringVar[Laskennallinen tekniikka]{Program}
\MakeStringVar[Lappeenranta]{City}

\MakeStringVar[Project]{Doctype}

\MakeStringVar{titlefin}
\MakeStringVar{Major}
\MakeStringVar{Majorfin}

\MakeStringVar{AcceptDate}
\MakeStringVar{SignDate}
\MakeStringVar{Year}
\MakeStringVar{FirstExaminer}

\MakeStringVar{Keywords}
\MakeStringVar[]{addtostats}

% defines commands \Examiner
\MakePeopleVar{Examiner}{Tarkastaja}{N.N.}
% examiners for the Finnish abstract only
\MakeStringVar{Tarkastajat}

\DeclareGraphicsExtensions{.pdf,.jpg,.png}

% A4 dimensions for the thesis.
\geometry{a4paper,tmargin=35mm,bmargin=20mm,lmargin=40mm,rmargin=20mm}

% Float caption style.
\captionsetup{labelfont=bf,labelsep=period,font=small}

% Subfloat caption style
\captionsetup[subfloat]{labelfont=normalfont,labelformat=parens}

% References in the order of cites.
\bibliographystyle{unsrt}

\renewcommand\maketitle{%
  \onehalfspacing%
  \setlength\parindent{0pt}%
  \setlength\parskip{0pt}%
  \thispagestyle{empty}%
  \begin{flushleft}
  
    \begin{figure} [!htp]
        \includegraphics{resources/LUT_University.png}
    \end{figure}

    \vspace{70mm}

    {\normalfont\large\bfseries\MakeUppercase{\@title}}

    \vspace{5mm}
    \vspace{\stretch{1}}

    \school

  \vspace{2.5mm}

  Foundations of Artificial Intelligence and Machine Learning, \EmitDoctype

  \vspace{2.5mm}
    
   \EmitYear\par
    
   \vspace{2.5mm}
    
   \@author\par
   
   \vspace{2.5mm}

  %% Tarkastaja: \EmitFirstExaminer
  
  \end{flushleft}%
}

\newcommand\personindent{3cm}

\newenvironment{personlist}[1]{%
  \begin{list}{}{%
    \renewcommand\makelabel[1]{#1\hfil}%
    \setlength\labelwidth{\personindent-\labelsep}%
    \setlength\leftmargin{\personindent}%
  }\item[#1]%
}{%
  \end{list}
}

\renewenvironment{abstract}{%
  \clearpage\thispagestyle{empty}%
  \setlength{\parskip}{1.6em}
  \section*{TIIVISTELMÄ}
  \school\\
  \EmitSchool\\
  \EmitProgram\par
  \@author\par
  \textbf{\@title}\par
  Harjoitustyö \\
  \EmitYear\\
  \Emitaddtostats
  \par
}{%
}




% Numbered page style for the main content
\newcommand\ps@masters{%
  \renewcommand\@oddfoot{}%
  \renewcommand\@evenfoot{}%
  \renewcommand\@oddhead{\hfil\normalfont\textrm{\thepage}}%
  \renewcommand\@evenhead{\normalfont\textrm{\thepage}\hfil}%
}

% Force a clearpage on every normal section
% \let\orig@section\section
% \renewcommand\section{\clearpage\orig@section}

% New page style for appendices.
\newcommand\ps@appendix{%
  \renewcommand\@oddfoot{\hfil\lastrightxmark}%
  \let\@evenfoot\@oddfoot%
  \renewcommand\@oddhead{%
    \normalfont\large\bfseries%
    Liite~\thesection.~\firstleftxmark\hfil}%
  \let\@evenhead\@oddhead%
}

\gdef\msc@app@title{muu}
\newcommand\msc@continues{}
\newcommand\msc@continued{(continued)}

% This command will start the appendices, and each appendix is a \section{}
% ended with \sectionend.
\renewcommand\appendix{%
  \clearpage%
  \pagestyle{appendix}%
  \renewcommand\thesection{\@arabic\c@section}%
  \renewcommand\thefigure{A\thesection.\@arabic\c@figure}%
  \renewcommand\thetable{A\thesection.\@arabic\c@table}%
  \setcounter{section}{0}%
  \setcounter{subsection}{0}%
  \addtocontents{toc}{\protect\contentsline{section}{LIITTEET}{}{}}%
  \renewcommand\section{%
    \clearpage%
    \setcounter{figure}{0}%
    \setcounter{table}{0}%
    \secdef\Appendix\sAppendix%
  }%
  \if@hyperref@enabled%
    \renewcommand\theHfigure{\theHsection.\arabic{figure}}%
    \renewcommand\theHtable{\theHsection.\arabic{table}}%
  \fi%
}

% the complex form to set appendix title
\newcommand\Appendix[2][?]{%
  \gdef\msc@app@title{#2}%
  \refstepcounter{section}%
  \addtocontents{toc}{\hspace*{1.5em}Liite~%
    \protect\makebox[1.5em][l]{\thesection:} #1\\}%
  %{\large\bfseries#2\par}%
  \sectionmark{#1}%
  \extramarks{\msc@app@title}{}%
  \extramarks{\msc@continued}{\msc@continues}%
  \@afterheading%
}

\newcommand\sectionend{%
  \FloatBarrier\null%
  \extramarks{\msc@continued}{}%
}

% the starred form (simple)
\newcommand\sAppendix[1]{%
  {\raggedleft\large\bfseries\appendixname\par #1\par}%
  \@afterheading\addvspace{\baselineskip}%
}

% Float separation spacings.
% The defaults:
% \setlength{\floatsep}{12pt plus 2pt minus 4pt}
% \setlength{\textfloatsep}{20pt plus 2pt minus 4pt}
% \setlength{\intextsep}{14pt plus 4pt minus 4pt}
% New settings:
\setlength{\intextsep}{26pt plus 6pt minus 2pt}
\setlength{\textfloatsep}{26pt plus 6pt minus 2pt}

